import os
import json
import random
import argparse
from pathlib import Path
from itertools import islice

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
import numpy as np
import wandb
from jaxtyping import Int, Float, Bool


class AutoEncoder(nn.Module):
    def __init__(
        self, d_hidden: int, l1_coeff: float, d_in: int, dtype=torch.float32, seed=47
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_in, d_hidden, dtype=dtype))
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, d_in, dtype=dtype))
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

    def forward(self, x: torch.Tensor):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct - x).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed


class Buffer():
    """
    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty. 
    """
    def __init__(self, cfg, model: HookedTransformer, all_tokens: Int[Tensor, "batch seq_len"]):
        self.buffer = torch.zeros((cfg["buffer_size"], cfg["d_mlp"]), dtype=torch.bfloat16, requires_grad=False).cuda()
        print(f"\nBuffer size: {self.buffer.shape}, batch_shape: {(cfg['batch_size'], cfg['d_mlp'])}")
        self.cfg = cfg
        self.token_pointer = 0
        self.first = True
        self.model = model
        self.hook_name = f'blocks.{cfg["layer"]}.{cfg["act"]}'
        self.all_tokens = all_tokens
        self.refresh()
    
    @torch.no_grad()
    def refresh(self):
        if self.first:
            num_batches = self.cfg["buffer_batches"]
            self.first = False
        else:
            num_batches = self.cfg["buffer_batches"]//2

        if (self.token_pointer + num_batches) >= self.all_tokens.shape[0]:
            return
        
        self.pointer = 0
        with torch.autocast("cuda", torch.bfloat16):
            for _ in range(0, num_batches, self.cfg["model_batch_size"]):
                tokens = self.all_tokens[self.token_pointer:self.token_pointer+self.cfg["model_batch_size"]]
                tokens = tokens.to(torch.long)
                _, cache = self.model.run_with_cache(tokens, names_filter=self.hook_name)
                mlp_acts = cache[self.hook_name].reshape(-1, self.cfg["d_mlp"])
                # print(tokens.shape, mlp_acts.shape, self.pointer, self.token_pointer, self.buffer[self.pointer: self.pointer+mlp_acts.shape[0]].shape)
                self.buffer[self.pointer: self.pointer+mlp_acts.shape[0]] = mlp_acts
                self.pointer += mlp_acts.shape[0]
                self.token_pointer += self.cfg["model_batch_size"]

        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).cuda()]

    @torch.no_grad()
    def __next__(self):
        if (self.pointer+self.cfg["batch_size"]) > self.buffer.shape[0]:
            raise StopIteration
        out = self.buffer[self.pointer:self.pointer+self.cfg["batch_size"]]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0]//2 - self.cfg["batch_size"]:
            #print("Refreshing the buffer!")
            self.refresh()
        return out
    
    def __iter__(self):
       return self

def get_german_prompt_data() -> Int[Tensor, "batch seq_len"]:
    """Token index tensors generated by the model being expanded"""
    os.makedirs("data/wikipedia", exist_ok=True)
    filenames = ["data/europarl/de_batched.pt"] + [
        f"data/wikipedia/{f}" for f in os.listdir("data/wikipedia") if f.endswith(".pt")
    ]
    return torch.cat([torch.load(filename) for filename in filenames], dim=0)


def main(model_name: str, layer: int, act_name: str, cfg: dict):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = HookedTransformer.from_pretrained(
        model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=device,
    )

    SEED = cfg["seed"]
    GENERATOR = torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_grad_enabled(True)

    d_model = model.cfg.d_model
    d_mlp = model.cfg.d_mlp
    cfg["d_mlp"] = d_mlp
    if act_name == "mlp.hook_post" or act_name == "mlp.hook_pre":
        d_in = d_mlp
    elif act_name == "hook_mlp_out":
        d_in = d_model
    else:
        raise ValueError(f"Unknown act_name: {act_name}")

    autoencoder_dim = d_in * cfg["expansion_factor"]
    encoder = AutoEncoder(
        autoencoder_dim, l1_coeff=cfg["l1_coeff"], d_in=d_in, seed=SEED
    ).to(device)
    encoder_optim = torch.optim.AdamW(
        encoder.parameters(),
        lr=cfg["lr"],
        betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=cfg["wd"],
    )

    prompt_data = get_german_prompt_data()
    num_tokens = torch.numel(prompt_data)
    num_batches = num_tokens // cfg["batch_size"]
    print(f"Total tokens: {num_tokens}, total batches: {num_batches}")

    if cfg["use_wandb"]:
        wandb.init(project="pythia_autoencoder", config=cfg)
        wandb_name = wandb.run.name
        save_name = f"{wandb_name.split('-')[-1]}_" + "_".join(wandb_name.split("-")[:-1])
    else:
        save_name = "local"
    Path(model_name).mkdir(exist_ok=True)
    with open(f"{model_name}/{save_name}.json", "w") as f:
        json.dump(cfg, f)

    @torch.no_grad()
    def resample_dead_directions(
        encoder: AutoEncoder,
        eval_batch: Float[Tensor, "batch d_mlp"],
        dead_directions: Bool[Tensor, "d_enc"],
        num_dead_directions: int,
    ):
        if num_dead_directions > 0:
            eval_batch = eval_batch.to(device)
            dead_direction_indices = torch.argwhere(dead_directions).flatten()
            _, x_reconstruct, _, _, _ = encoder(eval_batch)

            # Losses per batch item
            l2_loss = (x_reconstruct - eval_batch).pow(2).sum(-1)
            loss = l2_loss.pow(2)

            # Scale losses and sample
            loss_probs = loss / loss.sum()
            indices = torch.multinomial(
                loss_probs, num_dead_directions, replacement=False
            ).cpu()
            neuron_inputs = eval_batch[indices]
            neuron_inputs = F.normalize(neuron_inputs, dim=-1)  # batch d_mlp
            encoder.W_dec[dead_direction_indices, :] = neuron_inputs.clone().to(encoder.W_dec.dtype)

            # Encoder
            active_directions = torch.argwhere(~dead_directions).flatten()
            active_direction_norms = encoder.W_enc[:, active_directions].norm(
                p=2, dim=0
            )  # d_mlp d_active_dir
            mean_active_norm = active_direction_norms.mean() * 0.2
            newly_normalized_inputs = neuron_inputs * mean_active_norm

            encoder.W_enc[:, dead_direction_indices] = newly_normalized_inputs.T.clone().to(encoder.W_enc.dtype)
            encoder.b_enc[dead_direction_indices] = 0

    with tqdm(total=num_batches * cfg["epochs"], position=0, leave=True) as pbar:
        dead_directions = torch.ones(size=(autoencoder_dim,)).bool().to(device)
        for epoch in range(cfg["epochs"]):
            buffer = Buffer(cfg, model, prompt_data)
            num_eval_batches = 20
            eval_batch = torch.cat(
                list(islice(buffer, num_eval_batches)), dim=0
            )
            print(f"Eval batch shape: {eval_batch.shape}")
            for i, batch in enumerate(buffer):
                batch_index = ((num_batches - num_eval_batches) * epoch) + i
                loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(
                    batch.to(device)
                )
                loss.backward()
                encoder.remove_parallel_component_of_grads()
                encoder_optim.step()

                active_directions = (
                    (mid_acts != 0).sum(dim=-1).mean(dtype=torch.float32).item()
                )
                dead_directions = ((mid_acts != 0).sum(dim=0) == 0) & dead_directions
                num_dead_directions = dead_directions.sum().item()

                loss_dict = {
                    "batch": batch_index,
                    "loss": loss.item(),
                    "l2_loss": l2_loss.item(),
                    "l1_loss": l1_loss.item(),
                    "avg_directions": active_directions,
                    "dead_directions": num_dead_directions,
                }

                pbar.update(1)
                reset_steps = [25000, 50000, 75000, 100000]
                count_dead_direction_steps = [step - 12500 for step in reset_steps]

                if (batch_index + 1) in reset_steps:
                    resample_dead_directions(
                        encoder, eval_batch, dead_directions, num_dead_directions
                    )
                    print(f"\nResampled {num_dead_directions} dead directions")
                
                if (batch_index + 1) in count_dead_direction_steps + reset_steps:
                    print("Resetting dead direction counter")
                    dead_directions = (
                        torch.ones(size=(autoencoder_dim,)).bool().to(device)
                    )

                if (batch_index + 1) % 5000 == 0:
                    torch.save(encoder.state_dict(), f"{model_name}/{save_name}.pt")
                    print(
                        f"\n(Batch {batch_index}) Loss: {loss_dict['loss']:.2f}, L2 loss: {loss_dict['l2_loss']:.2f}, L1 loss: {loss_dict['l1_loss']:.2f}, Avg directions: {loss_dict['avg_directions']:.2f}, Dead directions: {num_dead_directions}"
                    )
                
                if cfg["use_wandb"]:
                    wandb.log(loss_dict)
                encoder_optim.zero_grad()
                del loss, x_reconstruct, mid_acts, l2_loss, l1_loss
    torch.save(encoder.state_dict(), f"{model_name}/{save_name}.pt")


def get_config():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    cfg = {
        "model": "pythia-70m",
        "layer": 5,
        "act": "mlp.hook_post",
        "expansion_factor": 8,
        "epochs": 1,
        "seed": 47,
        "lr": 1e-4,
        "l1_coeff": 1e-3,
        "wd": 1e-2,
        "beta1": 0.9,
        "beta2": 0.99,
        "batch_size": 512, # Tokens
        "buffer_mult": 256,
        "seq_len": 128,
        "use_wandb": True
    }

    # Buffer batches needs to be multiple of model batch size
    cfg["model_batch_size"] = cfg["batch_size"] // cfg["seq_len"] * 16
    cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
    cfg["buffer_batches"] = cfg["buffer_size"] // cfg["seq_len"] #(cfg["model_batch_size"] * cfg["seq_len"])

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
    return cfg


if __name__ == "__main__":
    cfg = get_config()
    main(cfg["model"], cfg["layer"], cfg["act"], cfg)
