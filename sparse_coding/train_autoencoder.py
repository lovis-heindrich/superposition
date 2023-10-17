import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import wandb
import argparse
from pathlib import Path
from itertools import islice
from torch import Tensor
from jaxtyping import Int, Float, Bool


class AutoEncoder(nn.Module):
    def __init__(self, d_hidden, l1_coeff, d_in, dtype=torch.float32, seed=47):
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
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj


def main(model_name: str, layer: int, act_name: str, expansion_factor: int, cfg: dict):
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

    german_data = torch.load("data/europarl/de_batched.pt")
    english_data = torch.load("data/europarl/en_batched.pt")

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
    if act_name == "mlp.hook_post" or act_name == "mlp.hook_pre":
        d_in = d_mlp
    elif act_name == "hook_mlp_out":
        d_in = d_model
    else:
        raise ValueError(f"Unknown act_name: {act_name}")
    
    @torch.no_grad()
    def get_batched_mlp_activations(
        prompt_data: list[Tensor], num_samples_per_batch=2
    ):
        iters = min([len(language_data) for language_data in prompt_data])
        for i in range(0, iters, num_samples_per_batch):
            batch = []
            for prompt_type in prompt_data:
                for j in range(num_samples_per_batch):
                    prompt = prompt_type[i+j].to(device)
                    _, cache = model.run_with_cache(
                        prompt, names_filter=f"blocks.{layer}.{act_name}"
                        )
                    acts = cache[f"blocks.{layer}.{act_name}"]
                    acts = acts.reshape(-1, d_in).cpu()
                    batch.append(acts)
            batch = torch.cat(batch, dim=0)
            random_order = torch.randperm(batch.shape[0], generator=GENERATOR)
            yield batch[random_order]
            

    autoencoder_dim = d_in * expansion_factor
    encoder = AutoEncoder(
        autoencoder_dim, l1_coeff=cfg["l1_coeff"], d_in=d_in, seed=SEED
    ).to(device)
    encoder_optim = torch.optim.AdamW(
        encoder.parameters(),
        lr=cfg["lr"],
        betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=cfg["wd"],
    )

    prompt_data = [german_data, english_data]
    num_samples_per_batch = 2
    num_batches = min([len(language_data) for language_data in prompt_data]) // num_samples_per_batch

    wandb.init(project="pythia_autoencoder", config=cfg)
    Path(model_name).mkdir(exist_ok=True)

    @torch.no_grad()
    def resample_dead_directions(encoder: AutoEncoder, eval_batch: Float[Tensor, "batch d_mlp"], dead_directions: Bool[Tensor, "d_enc"], num_dead_directions: int):
        if num_dead_directions > 0:
            eval_batch = eval_batch.to(device)
            dead_direction_indices = torch.argwhere(dead_directions).flatten()
            _, x_reconstruct, _, _, _ = encoder(eval_batch)

            # Losses per batch item
            l2_loss = (x_reconstruct - eval_batch).pow(2).sum(-1)
            loss = l2_loss.pow(2)

            # Scale losses and sample
            loss_probs = loss / loss.sum()
            indices = torch.multinomial(loss_probs, num_dead_directions, replacement=False).cpu()
            neuron_inputs = eval_batch[indices]
            neuron_inputs = F.normalize(neuron_inputs, dim=-1) # batch d_mlp
            encoder.W_dec[dead_direction_indices, :] = neuron_inputs.clone()

            # Encoder
            active_directions = torch.argwhere(~dead_directions).flatten()
            active_direction_norms = encoder.W_enc[:, active_directions].norm(p=2, dim=0) # d_mlp d_active_dir
            mean_active_norm = active_direction_norms.mean() * 0.2
            newly_normalized_inputs = neuron_inputs * mean_active_norm

            encoder.W_enc[:, dead_direction_indices] = newly_normalized_inputs.T.clone()
            encoder.b_enc[dead_direction_indices] = 0

    with tqdm(total=num_batches * cfg['epochs'], position=0, leave=True) as pbar:
        dead_directions = torch.ones(size=(autoencoder_dim,)).bool().to(device)
        for epoch in range(cfg['epochs']):
            batched_mlp_acts = get_batched_mlp_activations(
                prompt_data, num_samples_per_batch=num_samples_per_batch)
            num_eval_batches = 20
            eval_batch = torch.cat(list(islice(batched_mlp_acts, num_eval_batches)), dim=0)
            for i, batch in enumerate(batched_mlp_acts):
                batch_index = ((num_batches-num_eval_batches)*epoch)+i
                loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(batch.to(device))
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
                    "dead_directions": num_dead_directions
                }

                pbar.update(1)
                if (batch_index+1) in [2500, 10000, 20000, 30000]:
                    resample_dead_directions(encoder, eval_batch, dead_directions, num_dead_directions)
                    print(f"\nResampled {num_dead_directions} dead directions")

                if (batch_index+1) % 5000 == 0:
                    torch.save(encoder.state_dict(), f"{model_name}/{act_name}_l{layer}_{batch_index}.pt")
                    dead_directions = torch.ones(size=(autoencoder_dim,)).bool().to(device)
                    print(
                        f"\n(Batch {i}) Loss: {loss_dict['loss']:.2f}, L2 loss: {loss_dict['l2_loss']:.2f}, L1 loss: {loss_dict['l1_loss']:.2f}, Avg directions: {loss_dict['avg_directions']:.2f}, Dead directions: {num_dead_directions}"
                    )
                wandb.log(loss_dict)
                encoder_optim.zero_grad()
                del loss, x_reconstruct, mid_acts, l2_loss, l1_loss
    
    torch.save(encoder.state_dict(), f"{model_name}/{act_name}_l{layer}.pt")


def get_config():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    cfg = {
        "model": "pythia-70m",
        "layer": 5,
        "act": "hook_mlp_out",
        "expansion_factor": 4,
        "epochs": 2,
        "seed": 47,
        "lr": 1e-4,
        "l1_coeff": 5e-4,
        "wd": 1e-2,
        "beta1": 0.9,
        "beta2": 0.99,
    }

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
    main(cfg["model"], cfg["layer"], cfg["act"], cfg["expansion_factor"], cfg)
