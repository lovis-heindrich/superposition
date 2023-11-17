import json
import os
import sys
import argparse
import logging
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import plotly.io as pio
from jaxtyping import Int, Float
from torch import Tensor
import pandas as pd
sys.path.append('../')  # Add the parent directory to the system path
from process_tiny_stories_data import load_tinystories_validation_prompts
from pathlib import Path

pio.renderers.default = "notebook_connected"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)

logging.basicConfig(format='(%(levelname)s) %(asctime)s: %(message)s', level=logging.INFO, datefmt='%I:%M:%S')


import utils.haystack_utils as haystack_utils
from sparse_coding.train_autoencoder import AutoEncoder
from utils.autoencoder_utils import AutoEncoderConfig, load_encoder

from joblib import Memory
cachedir = '/workspace/cache'
data_path = '/workspace/'
os.makedirs(cachedir, exist_ok=True)
memory = Memory(cachedir, verbose=0, bytes_limit=20e9)


@torch.no_grad()
def get_acts(prompt: str, model: HookedTransformer, encoder: AutoEncoder, cfg: AutoEncoderConfig):
    _, cache = model.run_with_cache(prompt, names_filter=cfg.encoder_hook_point)
    acts = cache[cfg.encoder_hook_point].squeeze(0)
    _, _, mid_acts, _, _ = encoder(acts)
    return mid_acts


@memory.cache
def get_max_activations(prompts: tuple[str], model: HookedTransformer, encoder: AutoEncoder, cfg: AutoEncoderConfig):
    activations = []
    for prompt in tqdm(prompts):
        acts = get_acts(prompt, model, encoder, cfg)
        max_prompt_activation = acts.max(0)[0]
        activations.append(max_prompt_activation)

    max_activation_per_prompt = torch.stack(activations)  # n_prompt x d_enc

    total_activations = max_activation_per_prompt.sum(0)
    print(f"Active directions on validation data: {total_activations.nonzero().shape[0]} out of {total_activations.shape[0]}")
    return max_activation_per_prompt

@memory.cache
def load_cached_tiny_stories_prompts():
    return load_tinystories_validation_prompts()

def print_top_examples(model: HookedTransformer, encoder: AutoEncoder, cfg: AutoEncoderConfig, prompts: list[str], activations: Float[Tensor, "n_prompts d_enc"], direction: int, n=5):
    top_idxs = activations[:, direction].argsort(descending=True)[:n].cpu().tolist()
    for prompt_index in top_idxs:
        prompt = prompts[prompt_index]
        prompt_tokens = model.to_str_tokens(model.to_tokens(prompt))
        acts = get_acts(prompt, model, encoder, cfg)
        direction_act = acts[:, direction].cpu().tolist()
        max_direction_act = max(direction_act)
        if max_direction_act > 0:
            haystack_utils.color_print_strings(prompt_tokens, direction_act, max_value=max_direction_act)

@memory.cache
def load_model(model_name: str, device: str) -> HookedTransformer:
    return HookedTransformer.from_pretrained(
        model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=device,
    )

def main(model_name: str, run_name: str, label_file: str):
    if Path(label_file).is_file():
        overwrite = print("Model evaluation already exists. Overwrite? Y/n")
        if overwrite == 'n' or overwrite == 'N':
            return

    model = load_model(model_name, haystack_utils.get_device())

    encoder, cfg = load_encoder(run_name, data_path + model_name, model)
    prompts = load_cached_tiny_stories_prompts()
    max_activation_per_prompt = get_max_activations(tuple(prompts), model, encoder, cfg)

    labels = {label: [] for label in [1, 2, 3]}
    for direction in torch.randperm(len(encoder.W_dec))[:25]:
        print_top_examples(model, encoder, cfg, prompts, max_activation_per_prompt, direction, n=10)

        label = input(f'Enter 1 for ultra low density cluster, 2 for interpretable, and 3 for uninterpretable:')
        while label not in ['1', '2', '3']:
            label = input('Invalid input.')
        labels[int(label)].append(direction.item())

    with open(label_file, 'w') as f:
        json.dump(labels, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output",
        type=str,
    )
    args = parser.parse_args()

    os.makedirs(args.model, exist_ok=True)
    output = args.output or f"/workspace/data/{args.model}/{args.run}.json"
    os.makedirs(f"/workspace/data/{args.model}", exist_ok=True)
    
    main(args.model, args.run, output)
