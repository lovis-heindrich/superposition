import os
import sys

sys.path.append("../")
import setup

import json
import random
import argparse
from itertools import islice
from datasets import Dataset
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
from process_tiny_stories_data import (
    load_tinystories_validation_prompts,
    load_tinystories_tokens,
)
from utils.autoencoder_utils import (
    evaluate_autoencoder_reconstruction,
    train_autoencoder_evaluate_autoencoder_reconstruction,
    load_encoder,
    AutoEncoderConfig,
    act_name_to_d_in,
    get_component_baseline_losses
)
from utils.haystack_utils import get_device
from autoencoder import AutoEncoder
import time
from datasets import load_dataset


class Buffer:
    """
    This defines a data buffer, to store a bunch of model acts that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty.
    """

    def __init__(
        self,
        cfg,
        model: HookedTransformer,
        all_tokens: Int[Tensor, "batch seq_len"],
        device
    ):
        self.buffer = torch.zeros(
            (cfg["buffer_size"], cfg["d_in"]),
            dtype=torch.bfloat16,
            requires_grad=False,
        ).to(device)
        print(
            f"\nBuffer size: {self.buffer.shape}, batch_shape: {(cfg['batch_size'], cfg['d_in'])}"
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
                if len(self.buffer) <= self.pointer:
                    continue
                tokens = self.all_tokens[
                    self.token_pointer : self.token_pointer
                    + self.cfg["model_batch_size"]
                ].to(torch.long)
                with torch.no_grad():
                    def acts_hook(value, hook):
                        if "head_idx" in cfg:
                            hook.ctx["activation"] = value[:, :, cfg["head_idx"]].clone()
                        else:
                            hook.ctx["activation"] = value
                    acts_hooks = [(self.hook_name, acts_hook)]
                    with self.model.hooks(acts_hooks):
                        self.model(tokens)
                    
                    acts = self.model.hook_dict[self.hook_name].ctx['activation'].reshape(-1, self.cfg["d_in"])
                    truncated_acts = acts[: len(self.buffer) - self.pointer]
                    self.buffer[
                        self.pointer : self.pointer + acts.shape[0]
                    ] = truncated_acts
                    self.pointer += truncated_acts.shape[0]
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
    excess_kurt = torch.mean(((data - mean) / std) ** 4) - 3
    return mean.item(), variance.item(), skew.item(), excess_kurt.item()


def decoder_unembed_cosine_sim_mean_kurtosis(
    model: HookedTransformer, layer: int, W_dec: torch.Tensor
) -> torch.Tensor:
    """Return mean kurtosis over all decoder features' cosine sims with the unembed (higher is better)"""
    resid_dirs = F.normalize(W_dec @ model.W_out[layer], dim=-1)
    unembed = F.normalize(model.unembed.W_U, dim=0)
    cosine_sims = einops.einsum(
        resid_dirs, unembed, "d_hidden d_model, d_model d_vocab -> d_hidden d_vocab"
    )

    mean = cosine_sims.mean(dim=-1).unsqueeze(1)
    std = cosine_sims.std(dim=-1).unsqueeze(1)
    excess_kurt = torch.mean(((cosine_sims - mean) / std) ** 4) - 3
    return excess_kurt.mean()


def get_cosine_sim_moments(W_enc: torch.Tensor, W_dec: torch.Tensor):
    """Cosine similarity between corresponding features of the encoder and decoder weights"""
    cosine_sim = torch.nn.CosineSimilarity(dim=0)
    sims = cosine_sim(W_enc, W_dec.T)
    return get_moments(sims)

def get_encoder_sims(W_enc: torch.Tensor):
    normalized_W_enc = F.normalize(W_enc, dim=0)
    cosine_sims = (normalized_W_enc.T @ normalized_W_enc)
    mask = torch.tril(torch.ones_like(cosine_sims), diagonal=-1).flatten().bool()
    cosine_sims = cosine_sims.flatten()[mask].mean().item()
    return cosine_sims

def get_decoder_sims(W_dec: torch.Tensor):
    normalized_W_dec = W_dec
    cosine_sims = (normalized_W_dec @ normalized_W_dec.T)
    mask = torch.tril(torch.ones_like(cosine_sims), diagonal=-1).flatten().bool()
    cosine_sims = cosine_sims.flatten()[mask].mean().item()
    return cosine_sims

def get_entropy(activations: Float[Tensor, "batch d_hidden"]) -> float:
    """Entropy of the activations"""
    probs = F.normalize(activations, p=1, dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    return torch.mean(entropy).item()


def get_num_similar_dec_features(W_dec: Float[Tensor, "d_hidden d_component"]):
    normalized_W_dec = F.normalize(W_dec, dim=1)
    cosine_sims = torch.tril(normalized_W_dec @ normalized_W_dec.T, diagonal=-1).flatten()
    return cosine_sims


def main(
    encoder: AutoEncoder,
    model: HookedTransformer,
    cfg: dict,
    prompt_data: Int[Tensor, "n_examples seq_len"],
    eval_prompts: list[str],
    device: str
):
    encoder_optim = torch.optim.AdamW(
        encoder.parameters(),
        lr=cfg["lr"],
        betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=0.0,
    )

    hook_name = f"blocks.{cfg['layer']}.{cfg['act']}"
    baseline_loss, baseline_zero_ablation_loss = get_component_baseline_losses(eval_prompts, model, hook_name, disable_tqdm=True, prepend_bos=False)
    
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
    os.makedirs(f"{cfg['save_path']}/{cfg['model']}", exist_ok=True)
    with open(f"{cfg['save_path']}/{cfg['model']}/{save_name}.json", "w") as f:
        json.dump(cfg, f, indent=4)

    @torch.no_grad()
    def resample_dead_directions(
        encoder: AutoEncoder,
        eval_batch: Float[Tensor, "batch d_in"],
        direction_counter: Int[Tensor, "d_enc"],
        num_dead_directions: int,
        dead_direction_threshold: float
    ):
        if num_dead_directions > 0:
            dead_directions = (direction_counter <= dead_direction_threshold)
            dead_direction_indices = torch.argwhere(dead_directions).flatten()
            batch_reconstruct = []
            for i in range(
                0, cfg["num_eval_batches"] * cfg["batch_size"], cfg["batch_size"]
            ):
                _, x_reconstruct, _, _, _ = encoder(
                    eval_batch[i : i + cfg["batch_size"]].to(device)
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
            neuron_inputs = F.normalize(neuron_inputs, dim=-1)  # batch d_in
            encoder.W_dec[dead_direction_indices, :] = (
                neuron_inputs.clone().to(encoder.W_dec.dtype).to(device)
            )

            # Encoder
            active_directions = torch.argwhere(~dead_directions).flatten()
            active_direction_norms = (
                encoder.W_enc[:, active_directions].norm(p=2, dim=0).cpu()
            )  # d_in d_active_dir
            mean_active_norm = active_direction_norms.mean() * 0.2
            newly_normalized_inputs = neuron_inputs * mean_active_norm

            encoder.W_enc[:, dead_direction_indices] = (
                newly_normalized_inputs.T.clone().to(encoder.W_enc.dtype).to(device)
            )
            encoder.b_enc[dead_direction_indices] = 0

            encoder_optim.state_dict()["state"][0]["exp_avg"][:, dead_direction_indices] = 0
            encoder_optim.state_dict()["state"][0]["exp_avg_sq"][:, dead_direction_indices] = 0
            encoder_optim.state_dict()["state"][1]["exp_avg"][dead_direction_indices, :] = 0
            encoder_optim.state_dict()["state"][1]["exp_avg_sq"][dead_direction_indices, :] = 0
            encoder_optim.state_dict()["state"][2]["exp_avg"][dead_direction_indices] = 0
            encoder_optim.state_dict()["state"][2]["exp_avg_sq"][dead_direction_indices] = 0

    direction_counter = torch.zeros(size=(encoder.d_hidden,)).to(torch.int64).to(device)
    buffer = Buffer(cfg, model, prompt_data, device)
    eval_batch = torch.cat(list(islice(buffer, cfg["num_eval_batches"])), dim=0).detach().cpu()
    print(f"Eval batch shape: {eval_batch.shape}")

    reset_steps = [15000, 25000, 50000, 75000]
    reset_count_interval = 10000
    count_dead_direction_steps = [step - reset_count_interval for step in reset_steps]
    dead_direction_threshold = cfg["dead_direction_frequency"] * (reset_count_interval * cfg["batch_size"])
    eval_interval = 1000
    save_interval = 10000

    eval_direction_counter = torch.zeros(size=(encoder.d_hidden,)).to(torch.int64).to(device)
    eval_dead_direction_threshold = cfg["dead_direction_frequency"] * (eval_interval * cfg["batch_size"])

    for batch_index in tqdm(range(num_training_batches)):
        batch = next(buffer)
        loss, x_reconstruct, mid_acts, mse_loss, reg_losses = encoder(batch.to(device))
        loss.backward()
        encoder.remove_parallel_component_of_grads()
        encoder_optim.step()
        encoder.norm_decoder()

        active_directions = (
                    (mid_acts != 0).sum(dim=-1).mean(dtype=torch.float32).item()
                )

        #dead_directions = ((mid_acts != 0).sum(dim=0) == 0) & dead_directions
        count_active_directions = (mid_acts != 0).sum(dim=0)
        direction_counter += count_active_directions
        eval_direction_counter += count_active_directions

        if isinstance(reg_losses, list):
            l1_loss = reg_losses[0].item()
            hoyer_loss = reg_losses[1].item()
            loss_dict = {
            "batch": batch_index,
            "loss": loss.item(),
            "mse_loss": mse_loss.item(),
            "reg_loss": l1_loss + hoyer_loss,
            "l1_loss": l1_loss,
            "hoyer_loss": hoyer_loss,
            "epoch": buffer.epoch,
            "avg_directions": active_directions,
        }
        else:
            loss_dict = {
                "batch": batch_index,
                "loss": loss.item(),
                "mse_loss": mse_loss.item(),
                "reg_loss": reg_losses.item(),
                "epoch": buffer.epoch,
                "avg_directions": active_directions,
            }

        if ((batch_index + 1) % eval_interval == 0):
            start_time = time.time()
            with torch.no_grad():
                reconstruction_loss, fig = train_autoencoder_evaluate_autoencoder_reconstruction(
                    encoder,
                    cfg,
                    f'blocks.{cfg["layer"]}.{cfg["act"]}',
                    eval_prompts,
                    model,
                    reconstruction_loss_only=True,
                    show_tqdm=False,
                    prepend_bos=False
                )
                
                loss_recovered = ((baseline_zero_ablation_loss - reconstruction_loss)/(baseline_zero_ablation_loss - baseline_loss))
                #loss_recovered_mean = ((baseline_mean_ablation_loss - reconstruction_loss)/(baseline_mean_ablation_loss - baseline_loss))
                #print(baseline_zero_ablation_loss, baseline_mean_ablation_loss, reconstruction_loss, baseline_loss, loss_recovered, loss_recovered_mean)
                # b_mean, b_variance, b_skew, b_kurt = get_moments(encoder.b_enc)
                # (
                #     feature_cosine_sim_mean,
                #     feature_cosine_sim_variance,
                #     _,
                #     _,
                # ) = get_cosine_sim_moments(encoder.W_enc, encoder.W_dec)
                W_dec_norm = encoder.W_dec.norm()
                W_enc_norm = encoder.W_enc.norm()
                # entropy = get_entropy(mid_acts)
                
                # mean_feature_unembed_cosine_sim_kurtosis = (
                #     decoder_unembed_cosine_sim_mean_kurtosis(
                #         model, cfg["layer"], encoder.W_dec
                #     )
                # )
                # num_similar_dec_features = get_num_similar_dec_features(encoder.W_dec)

                encoder_sim = get_encoder_sims(encoder.W_enc)
                # decoder_sim = get_decoder_sims(encoder.W_dec)

                num_dead_directions = (eval_direction_counter <= eval_dead_direction_threshold).sum().item() #dead_directions.sum().item()
                eval_direction_counter = torch.zeros(size=(encoder.d_hidden,)).to(torch.int64).to(device)

                eval_dict = {
                    "reconstruction_loss": reconstruction_loss,
                    "loss_recovered": loss_recovered,
                    # "bias_mean": b_mean,
                    # "bias_std": b_variance**0.5,
                    # "bias_skew": b_skew,
                    # "bias_kurtosis": b_kurt,
                    # "feature_cosine_sim_mean": feature_cosine_sim_mean,
                    # "feature_cosine_sim_variance": feature_cosine_sim_variance,
                    # "mean_feature_unembed_cosine_sim_kurtosis": mean_feature_unembed_cosine_sim_kurtosis,
                    # "num_similar_dec_features": num_similar_dec_features,
                    "W_enc_norm": W_enc_norm,
                    "W_dec_norm": W_dec_norm,
                    # "entropy": entropy,
                    "dead_directions": num_dead_directions,
                    "feature_non_zero_act_means": fig,
                    "encoder_sim": encoder_sim,
                    # "decoder_sim": decoder_sim
                }

                print(f"Eval time: {time.time() - start_time:.2f} seconds")
                loss_dict.update(eval_dict)
                print(
                    f"\n(Batch {batch_index}) Loss: {loss_dict['loss']:.2f}, MSE loss: {loss_dict['mse_loss']:.2f}, reg_loss: {loss_dict['reg_loss']:.2f}, Avg directions: {loss_dict['avg_directions']:.2f}, Dead directions: {num_dead_directions}, Reconstruction loss: {reconstruction_loss:.2f}, Loss Recovered: {loss_recovered:.2f}"
                )
                
                    

        if (batch_index + 1) % save_interval == 0:
            if cfg["save_checkpoint_models"]:
                torch.save(
                    encoder.state_dict(), f"{cfg['model']}/{save_name}_{batch_index}.pt"
                )
            else:
                torch.save(encoder.state_dict(), f"{cfg['save_path']}/{cfg['model']}/{save_name}.pt")

            # Update L1 coefficient if aiming for a target number of active directions
            if (cfg["reg"] == "l1") and (cfg["l1_target"] != None):
                eps = 0.2
                update_coeff = 0.000002
                if active_directions > cfg["l1_target"] + eps:
                    new_coeff = encoder.reg_coeff + update_coeff
                elif active_directions < cfg["l1_target"] - eps:
                    new_coeff = encoder.reg_coeff - update_coeff
                if np.abs(active_directions - cfg["l1_target"]) > eps:
                    print(f"Updating reg coeff from {encoder.reg_coeff} to {new_coeff}")
                    encoder.reg_coeff = new_coeff
                    cfg["l1_coeff"] = encoder.reg_coeff
                    with open(f"{cfg['save_path']}/{cfg['model']}/{save_name}.json", "w") as f:
                        json.dump(cfg, f, indent=4)

        if (batch_index + 1) in reset_steps:
            num_dead_directions = (direction_counter <= dead_direction_threshold).sum().item() #dead_directions.sum().item()
            resample_dead_directions(
                encoder, eval_batch, direction_counter, num_dead_directions, dead_direction_threshold
            )
            loss_dict["resampled_directions"] = num_dead_directions
            print(f"\nResampled {num_dead_directions} dead directions")

        if (batch_index + 1) in count_dead_direction_steps + reset_steps:
            print("Resetting dead direction counter")
            direction_counter = torch.zeros(size=(encoder.d_hidden,)).to(device)


        if cfg["use_wandb"]:
            wandb.log(loss_dict)
        encoder_optim.zero_grad()
        del loss, x_reconstruct, mid_acts, mse_loss, reg_losses
    torch.save(encoder.state_dict(), f"{cfg['save_path']}/{cfg['model']}/{save_name}.pt")
    if cfg["use_wandb"]:
        wandb.finish()

    return reconstruction_loss, active_directions


DEFAULT_CONFIG = {
    "cfg_file": '',
    "data_path": "/workspace/data/tinystories",
    "save_path": "/workspace",
    "use_wandb": True,
    "num_eval_tokens": 800000,  # Tokens used to resample dead directions
    "num_training_tokens": 5e8,
    "batch_size": 5080,  # Batch shape is batch_size, d_in
    "buffer_mult": 256,  # Buffer size is batch_size*buffer_mult, d_in
    "seq_len": 127,
    "model": "tiny-stories-2L-33M",
    "layer": 0,
    "act": "mlp.hook_post",
    "expansion_factor": 4,
    "seed": 47,
    "lr": 1e-4,
    "l1_coeff": (0.0001, 0.00015),#(0.00011, 0.000165), #,  # Used for all regularization types to maintain backwards compatibility
    "l1_target": None,
    "wd": 1e-2,
    "beta1": 0.9,
    "beta2": 0.99,
    "num_eval_prompts": 150,  # Used for periodic evaluation logs
    "save_checkpoint_models": False,
    "reg": "combined_hoyer_sqrt", # l1 | sqrt | hoyer_square | hoyer_d | hoyer_d_scaled_l1 | combined_hoyer_l1 | combined_hoyer_sqrt
    "finetune_encoder": None,
    "dead_direction_frequency": 1e-5
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
    if cfg["cfg_file"]:
        with open(cfg["cfg_file"], "r") as f:
            cfg.update(json.load(f))

    # Derive config values
    cfg["model_batch_size"] = cfg["batch_size"] // cfg["seq_len"]
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
        autoencoder_dim = cfg['d_in'] * cfg["expansion_factor"]
        encoder = AutoEncoder(
            autoencoder_dim, reg_coeff=cfg["l1_coeff"], d_in=cfg['d_in'], seed=seed, reg=cfg["reg"]
        ).to(device)
        print(f"Input dim: {cfg['d_in']}, autoencoder dim: {autoencoder_dim}")
    return encoder


def get_dataset_dispatch(cfg) -> tuple[Dataset, Dataset]:
    if "tiny-stories" in cfg['model']:
        prompt_data = load_tinystories_tokens(cfg["data_path"], exclude_bos=True)
        eval_prompts = load_tinystories_validation_prompts(cfg["data_path"])[
            : cfg["num_eval_prompts"]
        ]
    elif "pythia" in cfg['model']:
        prompt_data = load_dataset("NeelNanda/pile-tokenized-2b", split='train')
        prompt_data.set_format(type="torch", columns=["tokens"])
        prompt_data = prompt_data["tokens"]
        
        eval_prompts = prompt_data[:cfg["num_eval_prompts"]].detach().cpu()
        prompt_data = prompt_data[cfg["num_eval_prompts"]:, 1:]
    else:
        raise ValueError(f"Model without registered training data: {cfg['model']}")
    return prompt_data, eval_prompts


def run_trial(parameters: dict):
    """Accepts parameters instead of command line arguments"""
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(parameters)
    # Accept alternative config values from file specified in command line
    if cfg["cfg_file"]:
        with open(cfg["cfg_file"], "r") as f:
            cfg.update(json.load(f))
    # Allow experiment to set coefficient values separately
    if cfg['reg_coeff_2']:
        cfg['l1_coeff'] = [cfg['l1_coeff'], cfg['reg_coeff_2']]

    # Derive config values
    cfg["model_batch_size"] = cfg["batch_size"] // cfg["seq_len"]
    cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
    cfg["buffer_batches"] = (
        cfg["buffer_size"] // cfg["seq_len"]
    )
    cfg["num_eval_batches"] = cfg["num_eval_tokens"] // cfg["batch_size"]
    assert (
        cfg["buffer_batches"] % cfg["model_batch_size"] == 0
    ), "Buffer batches must be multiple of model batch size"

    prompt_data, eval_prompts = get_dataset_dispatch(cfg)
    
    SEED = cfg["seed"]
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_grad_enabled(True)

    device = get_device()
    model = HookedTransformer.from_pretrained(
        cfg["model"],
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=device,
    )
    model.set_use_attn_result(True)

    cfg["d_in"] = act_name_to_d_in(model, cfg['act'])

    encoder = get_autoencoder(cfg, device, SEED)
    return main(encoder, model, cfg, prompt_data, eval_prompts, device)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    cfg = get_config()
    prompt_data, eval_prompts = get_dataset_dispatch(cfg)
    
    SEED = cfg["seed"]
    # GENERATOR = torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_grad_enabled(True)

    device = get_device()
    model = HookedTransformer.from_pretrained(
        cfg["model"],
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=device,
    )
    model.set_use_attn_result(True)

    cfg["d_in"] = act_name_to_d_in(model, cfg['act'])

    encoder = get_autoencoder(cfg, device, SEED)
    main(encoder, model, cfg, prompt_data, eval_prompts, device)
