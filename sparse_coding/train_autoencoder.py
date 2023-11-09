import os
import sys

sys.path.append("../")
import json
import random
import argparse
from itertools import islice

import torch
import einops
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
import numpy as np
import wandb
from jaxtyping import Int, Float, Bool
from process_tiny_stories_data import load_tinystories_validation_prompts, load_tinystories_tokens
from utils.autoencoder_utils import evaluate_autoencoder_reconstruction, load_encoder, AutoEncoderConfig
from utils.haystack_utils import get_occurring_tokens
from autoencoder import AutoEncoder
import time

def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

class Buffer:
    """
    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty.
    """

    def __init__(
        self,
        cfg,
        model: HookedTransformer,
        all_tokens: Int[Tensor, "batch seq_len"],
        device=get_device(),
    ):
        self.buffer = torch.zeros(
            (cfg["buffer_size"], cfg["d_mlp"]),
            dtype=torch.bfloat16,
            requires_grad=False,
        ).to(device)
        print(
            f"\nBuffer size: {self.buffer.shape}, batch_shape: {(cfg['batch_size'], cfg['d_mlp'])}"
        )
        self.cfg = cfg
        self.token_pointer = 0
        self.first = True
        self.model = model
        self.hook_name = f'blocks.{cfg["layer"]}.{cfg["act"]}'
        self.all_tokens = all_tokens
        self.device = device
        self.epoch = 0
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        if self.first:
            num_batches = self.cfg["buffer_batches"]
            self.first = False
        else:
            num_batches = self.cfg["buffer_batches"] // 2

        # If all data is used this might result in some data being reused in the very last buffer used in training
        # available_batches = self.all_tokens.shape[0] - self.token_pointer - 1
        # num_batches = min(num_batches, available_batches)

        # if num_batches <= 0:
        #     return

        self.pointer = 0
        with torch.autocast("cuda", torch.bfloat16):
            for _ in range(0, num_batches, self.cfg["model_batch_size"]):
                if (
                    self.token_pointer + self.cfg["model_batch_size"]
                    > self.all_tokens.shape[0]
                ):
                    self.token_pointer = 0
                    self.epoch += 1
                    print("Resetting token pointer")
                tokens = self.all_tokens[
                    self.token_pointer : self.token_pointer
                    + self.cfg["model_batch_size"]
                ]
                tokens = tokens.to(torch.long)
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(
                        tokens, names_filter=self.hook_name
                    )
                mlp_acts = cache[self.hook_name].reshape(-1, self.cfg["d_mlp"])
                # print(tokens.shape, mlp_acts.shape, self.pointer, self.token_pointer, self.buffer[self.pointer: self.pointer+mlp_acts.shape[0]].shape)
                if len(self.buffer) - self.pointer > 0:
                    truncated_mlp_acts = mlp_acts[: len(self.buffer) - self.pointer]
                    self.buffer[
                        self.pointer : self.pointer + mlp_acts.shape[0]
                    ] = truncated_mlp_acts
                    self.pointer += truncated_mlp_acts.shape[0]
                    self.token_pointer += self.cfg["model_batch_size"]

        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to(self.device)]

    @torch.no_grad()
    def __next__(self):
        if (self.pointer + self.cfg["batch_size"]) > self.buffer.shape[0]:
            raise StopIteration
        out = self.buffer[self.pointer : self.pointer + self.cfg["batch_size"]]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0] // 2 - self.cfg["batch_size"]:
            # print("Refreshing the buffer!")
            self.refresh()
        return out

    def __iter__(self):
        return self

def get_moments(data: torch.Tensor) -> tuple[float, float, float, float]:    
    mean = torch.mean(data)
    variance = torch.mean((data - mean) ** 2)
    std = torch.sqrt(variance)
    skew = torch.mean(((data - mean) / std) ** 3)
    kurt = torch.mean(((data - mean) / std) ** 4) - 3
    return mean.item(), variance.item(), skew.item(), kurt.item()


def decoder_unembed_cosine_sim_mean_kurtosis(model: HookedTransformer, layer: int, W_dec: torch.Tensor) -> torch.Tensor:
    '''Return mean kurtosis over all decoder features' cosine sims with the unembed (higher is better)'''
    W_out = model.W_out[layer]
    resid_dirs = torch.nn.functional.normalize(W_dec @ W_out, dim=-1)
    unembed = torch.nn.functional.normalize(model.unembed.W_U, dim=0)
    cosine_sims = einops.einsum(resid_dirs, unembed, 'd_hidden d_model, d_model d_vocab -> d_hidden d_vocab')
    
    vocab_occurs = get_occurring_tokens(model, tuple(load_tinystories_validation_prompts()))
    cosine_sims = cosine_sims[:, vocab_occurs == 1]
    
    mean = einops.repeat(cosine_sims.mean(dim=-1), f'd_hidden -> d_hidden {cosine_sims.shape[1]}')
    std = einops.repeat(cosine_sims.std(dim=-1), f'd_hidden -> d_hidden {cosine_sims.shape[1]}')
    kurt = torch.mean((cosine_sims - mean / std) ** 4) - 3

    return kurt.mean()

def get_cosine_sim_moments(W_enc: torch.Tensor, W_dec: torch.Tensor):
    """Cosine similarity between corresponding features of the encoder and decoder weights"""
    cosine_sim = torch.nn.CosineSimilarity(dim=1)
    sims = cosine_sim(W_enc, W_dec.T)
    mean = torch.mean(sims)
    variance = torch.mean((sims - mean) ** 2)
    std = torch.sqrt(variance)
    skew = torch.mean(((sims - mean) / std) ** 3)
    kurt = torch.mean(((sims - mean) / std) ** 4) - 3
    return mean.item(), variance.item(), skew.item(), kurt.item()

def get_entropy(activations: Float[Tensor, "batch d_enc"]) -> float:
    """Entropy of the activations"""
    probs = F.normalize(activations, p=1, dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    return torch.mean(entropy).item()


def main(encoder: AutoEncoder, model: HookedTransformer, cfg: dict, prompt_data: Int[Tensor, "n_examples seq_len"], eval_prompts: list[str]):

    encoder_optim = torch.optim.AdamW(
        encoder.parameters(),
        lr=cfg["lr"],
        betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=0.0
    )

    num_tokens = torch.numel(prompt_data)
    num_eval_tokens = cfg["num_eval_batches"] * cfg["batch_size"]

    if num_tokens < (num_eval_tokens + cfg["buffer_size"]):
        raise ValueError(
            f"Not enough tokens for training: {num_tokens} < {num_eval_tokens} + {cfg['buffer_size']}"
        )

    num_total_tokens = cfg["num_training_tokens"] + num_eval_tokens
    num_training_tokens = int(num_total_tokens - num_eval_tokens)
    num_training_batches = num_training_tokens // cfg["batch_size"]
    print(
        f"Total tokens: {num_tokens, num_eval_tokens, num_training_tokens}, total batches: {num_training_batches}"
    )

    if cfg["use_wandb"]:
        wandb.init(project=f'{cfg["model"]}-{cfg["layer"]}-autoencoder', config=cfg)
        wandb_name = wandb.run.name
        save_name = f"{wandb_name.split('-')[-1]}_" + "_".join(
            wandb_name.split("-")[:-1]
        )
        cfg["wandb_name"] = wandb_name
    else:
        save_name = "local"
    cfg["save_name"] = save_name
    os.makedirs(cfg['model'], exist_ok=True)
    # Path(cfg['model']).mkdir(exist_ok=True)
    with open(f"{cfg['model']}/{save_name}.json", "w") as f:
        json.dump(cfg, f, indent=4)

    @torch.no_grad()
    def resample_dead_directions(
        encoder: AutoEncoder,
        eval_batch: Float[Tensor, "batch d_mlp"],
        dead_directions: Bool[Tensor, "d_enc"],
        num_dead_directions: int,
    ):
        if num_dead_directions > 0:
            dead_direction_indices = torch.argwhere(dead_directions).flatten()
            batch_reconstruct = []
            for i in range(
                0, cfg["num_eval_batches"] * cfg["batch_size"], cfg["batch_size"]
            ):
                _, x_reconstruct, _, _, _ = encoder(
                    eval_batch[i : i + cfg["batch_size"]].to(get_device())
                )
                batch_reconstruct.append(x_reconstruct)
            x_reconstruct = torch.cat(batch_reconstruct, dim=0).cpu()

            # Losses per batch item
            l2_loss = (x_reconstruct - eval_batch).pow(2).sum(-1)
            loss = l2_loss.pow(2)

            # Scale losses and sample
            loss_probs = loss / loss.sum()
            indices = torch.multinomial(
                loss_probs, num_dead_directions, replacement=False
            )
            neuron_inputs = eval_batch[indices]
            neuron_inputs = F.normalize(neuron_inputs, dim=-1)  # batch d_mlp
            encoder.W_dec[dead_direction_indices, :] = (
                neuron_inputs.clone().to(encoder.W_dec.dtype).to(device)
            )

            # Encoder
            active_directions = torch.argwhere(~dead_directions).flatten()
            active_direction_norms = encoder.W_enc[:, active_directions].norm(
                p=2, dim=0
            ).cpu()  # d_mlp d_active_dir
            mean_active_norm = active_direction_norms.mean() * 0.2
            newly_normalized_inputs = neuron_inputs * mean_active_norm

            encoder.W_enc[:, dead_direction_indices] = (
                newly_normalized_inputs.T.clone().to(encoder.W_enc.dtype).to(device)
            )
            encoder.b_enc[dead_direction_indices] = 0

            encoder_optim.state_dict()["state"][0]["exp_avg"][
                :, dead_direction_indices
            ] = 0
            encoder_optim.state_dict()["state"][0]["exp_avg_sq"][
                :, dead_direction_indices
            ] = 0
            encoder_optim.state_dict()["state"][1]["exp_avg"][
                dead_direction_indices, :
            ] = 0
            encoder_optim.state_dict()["state"][1]["exp_avg_sq"][
                dead_direction_indices, :
            ] = 0
            encoder_optim.state_dict()["state"][2]["exp_avg"][
                dead_direction_indices
            ] = 0
            encoder_optim.state_dict()["state"][2]["exp_avg_sq"][
                dead_direction_indices
            ] = 0


    dead_directions = torch.ones(size=(encoder.d_hidden,)).bool().to(device)
    buffer = Buffer(cfg, model, prompt_data)
    eval_batch = torch.cat(list(islice(buffer, cfg["num_eval_batches"])), dim=0).cpu()
    print(f"Eval batch shape: {eval_batch.shape}")
    
    reset_steps = [25000, 50000, 75000, 100000]
    count_dead_direction_steps = [step - 12500 for step in reset_steps]
    dead_directions_reset_interval = 50000
    eval_interval = 1000
    save_interval = 10000

    for batch_index in tqdm(range(num_training_batches)):
        batch = next(buffer)
        loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(batch.to(device))
        loss.backward()
        encoder.remove_parallel_component_of_grads()
        encoder_optim.step()
        
        dead_directions = ((mid_acts != 0).sum(dim=0) == 0) & dead_directions
        
        loss_dict = {
            "batch": batch_index,
            "loss": loss.item(),
            "l2_loss": l2_loss.item(),
            "l1_loss": l1_loss.item(),
            "epoch": buffer.epoch
        }

        if (batch_index + 1) % eval_interval == 0:
            reconstruction_loss = evaluate_autoencoder_reconstruction(encoder, f'blocks.{cfg["layer"]}.{cfg["act"]}', eval_prompts, model, reconstruction_loss_only=True, show_tqdm=False)
            b_mean, b_variance, b_skew, b_kurt = get_moments(encoder.b_enc)
            feature_cosine_sim_mean, feature_cosine_sim_variance, _, _  = get_cosine_sim_moments(encoder.W_enc, encoder.W_dec)
            W_dec_norm = encoder.W_dec.norm()
            W_enc_norm = encoder.W_enc.norm()
            entropy = get_entropy(mid_acts)
            active_directions = (mid_acts != 0).sum(dim=-1).mean(dtype=torch.float32).item()
            num_dead_directions = dead_directions.sum().item()
            mean_feature_unembed_cosine_sim_kurtosis = decoder_unembed_cosine_sim_mean_kurtosis(model, cfg["layer"], encoder.W_dec)

            eval_dict = {
                "reconstruction_loss": reconstruction_loss,
                "bias_mean": b_mean,
                "bias_std": b_variance ** .5,
                "bias_skew": b_skew,
                "bias_kurtosis": b_kurt,
                "feature_cosine_sim_mean": feature_cosine_sim_mean,
                "feature_cosine_sim_variance": feature_cosine_sim_variance,
                "mean_feature_unembed_cosine_sim_kurtosis": mean_feature_unembed_cosine_sim_kurtosis,
                "W_enc_norm": W_enc_norm,
                "W_dec_norm": W_dec_norm,
                "entropy": entropy,
                "avg_directions": active_directions,
                "dead_directions": num_dead_directions,
            }
            loss_dict.update(eval_dict)
            print(
                f"\n(Batch {batch_index}) Loss: {loss_dict['loss']:.2f}, L2 loss: {loss_dict['l2_loss']:.2f}, L1 loss: {loss_dict['l1_loss']:.2f}, Avg directions: {loss_dict['avg_directions']:.2f}, Dead directions: {num_dead_directions}, Reconstruction loss: {reconstruction_loss:.2f}"
            )

        if (batch_index + 1) % save_interval == 0:
            if cfg["save_checkpoint_models"]:
                torch.save(encoder.state_dict(), f"{cfg['model']}/{save_name}_{batch_index}.pt")
            else:
                torch.save(encoder.state_dict(), f"{cfg['model']}/{save_name}.pt")

        if (batch_index + 1) in reset_steps:
            num_dead_directions = dead_directions.sum().item()
            resample_dead_directions(
                encoder, eval_batch, dead_directions, num_dead_directions
            )
            print(f"\nResampled {num_dead_directions} dead directions")

        if (batch_index + 1) in count_dead_direction_steps + reset_steps:
            print("Resetting dead direction counter")
            dead_directions = torch.ones(size=(encoder.d_hidden,)).bool().to(device)

        if ((batch_index + 1) > max(reset_steps)) and (
            (batch_index + 1) % dead_directions_reset_interval == 0
        ):
            dead_directions = torch.ones(size=(encoder.d_hidden,)).bool().to(device)

        if cfg["use_wandb"]:
            wandb.log(loss_dict)
        encoder_optim.zero_grad()
        del loss, x_reconstruct, mid_acts, l2_loss, l1_loss
    torch.save(encoder.state_dict(), f"{cfg['model']}/{save_name}.pt")
    if cfg["use_wandb"]:
        wandb.finish()


DEFAULT_CONFIG = {
    "cfg_file": None,
    "data_path": "data/tinystories",
    "use_wandb": True,
    "num_eval_tokens": 800000,  # Tokens used to resample dead directions
    "num_training_tokens": 5e8,
    "batch_size": 4096,  # Batch shape is batch_size, d_mlp
    "buffer_mult": 128,  # Buffer size is batch_size*buffer_mult, d_mlp
    "seq_len": 128,
    "model": "tiny-stories-2L-33M",
    "layer": 1,
    "act": "mlp.hook_post",
    "expansion_factor": 4,
    "seed": 47,
    "lr": 3e-5,
    "l1_coeff": 3e-6, # Used for both square root and L1 regularization to maintain backwards compatibility
    "wd": 1e-2,
    "beta1": 0.9,
    "beta2": 0.99,
    "num_eval_prompts": 200, # Used for periodic evaluation logs
    "save_checkpoint_models": False,
    "use_sqrt_reg": True,
    "finetune_encoder": None
}

def get_config():
    cfg = DEFAULT_CONFIG.copy()

    # Accept alternative config values from command line
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for key, value in cfg.items():
        if type(value) == bool:
            # argparse for Booleans is broken rip. Now you put in a flag to change the default --{flag} to set True, --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)

    # Accept alternative config values from file specified in command line
    if cfg["cfg_file"] is not None:
        with open(cfg["cfg_file"], "r") as f:
            cfg.update(json.load(f))

    # Derive config values
    cfg["model_batch_size"] = cfg["batch_size"] // cfg["seq_len"] * 16
    cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
    cfg["buffer_batches"] = (
        cfg["buffer_size"] // cfg["seq_len"]
    )  # (cfg["model_batch_size"] * cfg["seq_len"])
    cfg["num_eval_batches"] = cfg["num_eval_tokens"] // cfg["batch_size"]
    assert (
        cfg["buffer_batches"] % cfg["model_batch_size"] == 0
    ), "Buffer batches must be multiple of model batch size"

    return cfg


def get_autoencoder(cfg: AutoEncoderConfig, device: str, seed: int):
    if cfg["finetune_encoder"] is not None:
        encoder, encoder_cfg = load_encoder(cfg["finetune_encoder"], cfg["model"], model)
        assert encoder_cfg.layer == cfg["layer"]
        assert encoder_cfg.act_name == cfg["act"]
        assert encoder_cfg.expansion_factor == cfg["expansion_factor"]
        encoder.reg_coeff = cfg["l1_coeff"]
    else:
        if  cfg["act"] == "mlp.hook_post" or  cfg["act"] == "mlp.hook_pre":
            d_in = cfg["d_mlp"]
        elif  cfg["act"] == "hook_mlp_out":
            d_in = model.cfg.d_model
        else:
            raise ValueError(f"Unknown  act_name: {cfg['act']}")

        autoencoder_dim = d_in * cfg["expansion_factor"]
        encoder = AutoEncoder(
            autoencoder_dim, reg_coeff=cfg["l1_coeff"], d_in=d_in, seed=seed, reg="sqrt" if cfg["use_sqrt_reg"] else "l1"
        ).to(device)
        print(f"Input dim: {d_in}, autoencoder dim: {autoencoder_dim}")
    return encoder


if __name__ == "__main__":
    torch.cuda.empty_cache()
    cfg = get_config()
    prompt_data = load_tinystories_tokens(cfg["data_path"])
    eval_prompts = load_tinystories_validation_prompts(cfg["data_path"])[:cfg["num_eval_prompts"]]

    SEED = cfg["seed"]
    # GENERATOR = torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_grad_enabled(True)

    device = get_device()
    model = HookedTransformer.from_pretrained(
        cfg["model"],
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=device,
    )

    cfg["d_mlp"] = model.cfg.d_mlp

    encoder = get_autoencoder(cfg, device, SEED)

    # for l1_coeff in [0.00001, 0.0001, 0.0005, 0.001, 0.01]:
    #     torch.cuda.empty_cache()
    #     cfg["l1_coeff"] = l1_coeff
    #     main(cfg["model"], cfg["act"], cfg, prompt_data, eval_prompts)
    main(encoder, model, cfg, prompt_data, eval_prompts)
