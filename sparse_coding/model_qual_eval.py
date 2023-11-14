import re
import json
import pickle
import os
import sys
import argparse
import logging
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
from pathlib import Path
from jaxtyping import Int, Float
from torch import Tensor
import einops
from collections import Counter
from datasets import load_dataset
import pandas as pd
from ipywidgets import interact, IntSlider
sys.path.append('../')  # Add the parent directory to the system path
from process_tiny_stories_data import load_tinystories_validation_prompts, load_tinystories_tokens
from typing import Literal
from utils.plotting_utils import line

pio.renderers.default = "notebook_connected"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)

logging.basicConfig(format='(%(levelname)s) %(asctime)s: %(message)s', level=logging.INFO, datefmt='%I:%M:%S')


import utils.haystack_utils as haystack_utils
from sparse_coding.train_autoencoder import AutoEncoder
from utils.autoencoder_utils import custom_forward, AutoEncoderConfig, evaluate_autoencoder_reconstruction, get_encoder_feature_frequencies, load_encoder

from joblib import Memory
cachedir = './workspace/cache'
os.makedirs(cachedir, exist_ok=True)
memory = Memory(cachedir, verbose=0, bytes_limit=20e9)


@torch.no_grad()
def get_acts(prompt: str, model: HookedTransformer, encoder: AutoEncoder, cfg: AutoEncoderConfig):
    _, cache = model.run_with_cache(prompt, names_filter=cfg.encoder_hook_point)
    acts = cache[cfg.encoder_hook_point].squeeze(0)
    _, _, mid_acts, _, _ = encoder(acts)
    return mid_acts


@memory.cache
def get_max_activations(prompts: list[str], model: HookedTransformer, encoder: AutoEncoder, cfg: AutoEncoderConfig):
    activations = []
    for prompt in tqdm(prompts):
        acts = get_acts(prompt, model, encoder, cfg)
        max_prompt_activation = acts.max(0)[0]
        activations.append(max_prompt_activation)

    max_activation_per_prompt = torch.stack(activations)  # n_prompt x d_enc

    total_activations = max_activation_per_prompt.sum(0)
    print(f"Active directions on validation data: {total_activations.nonzero().shape[0]} out of {total_activations.shape[0]}")
    return max_activation_per_prompt


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


def main(model_name: str, run_name: str, label_path: str):
    model = HookedTransformer.from_pretrained(
        model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=haystack_utils.get_device(),
    )

    encoder, cfg = load_encoder(run_name, model_name, model)
    prompts = load_tinystories_validation_prompts()
    max_activation_per_prompt = get_max_activations(prompts, model, encoder, cfg)

    labels = {label: [] for label in [1, 2, 3]}
    for direction in torch.randperm(len(encoder.W_dec))[:25]:
        print_top_examples(model, encoder, cfg, prompts, max_activation_per_prompt, direction, n=10)

        label = input(f'Enter 1 for ultra low density cluster, 2 for interpretable, and 3 for uninterpretable:')
        while label not in ['1', '2', '3']:
            label = input('Invalid input.')
        labels[int(label)].append(direction.item())

    with open(label_path, 'w') as f:
        json.dump(labels, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--label_path",
        type=str,
    )
    args = parser.parse_args()

    os.makedirs(args.model_name, exist_ok=True)
    label_path = args.label_path or f"data/{args.model_name}/{args.run_name}"
    
    main(args.model_name, args.run_name, label_path)
