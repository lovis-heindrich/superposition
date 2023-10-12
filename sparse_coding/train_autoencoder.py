import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import plotly.io as pio
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import wandb
import plotly.express as px
import pandas as pd
import torch.nn.init as init
import argparse

import sys
sys.path.append('../')  # Add the parent directory to the system path
import context_neuron.haystack_utils as haystack_utils


def log(data):
    try:
        wandb.log(data)
    except Exception as e:
        print("Failed to log to W&B:", e)
        pass


class AutoEncoder(nn.Module):
    def __init__(self, d_hidden, l1_coeff, d_mlp, dtype=torch.float32, seed=47):
        super().__init__()
        torch.manual_seed(seed)
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_mlp, d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, d_mlp, dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_mlp, dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff
    
    def forward(self, x):
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
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj


def main(model_name: str, layer: int):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=device)

    german_data = haystack_utils.load_json_data("data/german_europarl.json")
    english_data = haystack_utils.load_json_data("data/english_europarl.json")

    cfg = {
        "seed": 47,
        "batch_size": 32,
        "model_batch_size": 128,
        "lr": 1e-4,
        "num_tokens": int(1e7),
        "l1_coeff": 3e-3,
        "wd": 1e-2,
        "beta1": 0.9,
        "beta2": 0.99,
        "dict_mult": 0.5,
        "seq_len": 128,
    }
    cfg["model_batch_size"] = cfg["batch_size"] // cfg["seq_len"] * 16

    SEED = cfg["seed"]
    GENERATOR = torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_grad_enabled(True)


    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    d_mlp = model.cfg.d_mlp
    d_vocab = model.cfg.d_vocab


    @torch.no_grad()
    def get_mlp_acts(prompts, layer=layer):
        acts = []
        for prompt in prompts:
            _, cache = model.run_with_cache(prompt, names_filter=f"blocks.{layer}.mlp.hook_post")
            mlp_acts = cache[f"blocks.{layer}.mlp.hook_post"]
            mlp_acts = mlp_acts.reshape(-1, d_mlp).cpu()
            acts.append(mlp_acts)
        return torch.cat(acts, dim=0)

    def get_batched_mlp_activations(prompt_data: list[list[str]], num_prompts_per_batch=10):
        iters = min([len(language_data) for language_data in prompt_data])
        for i in range(0, iters, num_prompts_per_batch):
            data = []
            for language_data in prompt_data:
                data.extend(language_data[i:i+num_prompts_per_batch])
            acts = get_mlp_acts(data)
            random_order = torch.randperm(acts.shape[0], generator=GENERATOR)
            yield acts[random_order]


    l1_coeff = 0.01
    expansion_factor = 4

    autoencoder_dim = d_mlp * expansion_factor
    encoder = AutoEncoder(autoencoder_dim, l1_coeff=cfg['l1_coeff'], d_mlp=d_mlp, seed=SEED).to(device)
    encoder_optim = torch.optim.AdamW(encoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]), weight_decay=cfg["wd"])


    prompt_data = [german_data, english_data]
    num_prompts_per_batch = 2
    num_batches =  min([len(language_data) for language_data in prompt_data]) // num_prompts_per_batch
    print(num_batches)
    batched_mlp_acts = get_batched_mlp_activations(prompt_data, num_prompts_per_batch=num_prompts_per_batch)


    wandb.init(project="pythia_autoencoder")


    batched_mlp_acts = get_batched_mlp_activations(prompt_data, num_prompts_per_batch=num_prompts_per_batch)
    with tqdm(total=num_batches) as pbar:
        dead_directions = torch.ones(size=(autoencoder_dim,)).bool().to(device)
        for i, batch in enumerate(batched_mlp_acts):
            loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(batch.to(device))
            loss.backward()
            encoder.remove_parallel_component_of_grads()
            encoder_optim.step()
            encoder_optim.zero_grad()

            active_directions = (mid_acts != 0).sum(dim=-1).mean(dtype=torch.float32).item()
            dead_directions = ((mid_acts != 0).sum(dim=0) == 0) & dead_directions

            loss_dict = {"batch": batch, "loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item(), "avg_directions": active_directions}

            pbar.update(1)
            if i % 250 == 0:
                num_dead_directions = dead_directions.sum().item()
                dead_direction_indices = torch.argwhere(dead_directions).flatten()
                init.kaiming_uniform_(encoder.W_enc[:, dead_direction_indices])
                init.kaiming_uniform_(encoder.W_dec[dead_direction_indices, :])
                dead_directions = torch.ones(size=(autoencoder_dim,)).bool().to(device)            
                print(f"(Batch {i}) Loss: {loss_dict['loss']:.2f}, L2 loss: {loss_dict['l2_loss']:.2f}, L1 loss: {loss_dict['l1_loss']:.2f}, Avg directions: {loss_dict['avg_directions']:.2f}, Dead directions: {num_dead_directions}")
                loss_dict["dead_directions"] = num_dead_directions
            log(loss_dict)
            del loss, x_reconstruct, mid_acts, l2_loss, l1_loss


    torch.save(encoder.state_dict(), f'{model_name}/output_mlp_l{layer}.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="pythia-70m",
        help="Name of model from TransformerLens",
    )
    parser.add_argument("--layer", default="5")

    args = parser.parse_args()

    main(args.model, args.layer)



