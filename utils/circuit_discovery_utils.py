import re
import json
import pickle
import os
import sys
import requests
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
from process_tiny_stories_data import load_tinystories_validation_prompts, load_tinystories_tokens
from typing import Literal
from transformer_lens.utils import test_prompt

from sparse_coding.train_autoencoder import AutoEncoder
from utils.autoencoder_utils import custom_forward, AutoEncoderConfig, evaluate_autoencoder_reconstruction, get_encoder_feature_frequencies, load_encoder, generate_with_encoder
import utils.haystack_utils as haystack_utils
from utils.plotting_utils import line


@torch.no_grad()
def get_acts(prompt: str | Tensor, model: HookedTransformer, encoder: AutoEncoder, cfg: AutoEncoderConfig):
    _, cache = model.run_with_cache(prompt, names_filter=cfg.encoder_hook_point)
    acts = cache[cfg.encoder_hook_point].squeeze(0)
    _, _, mid_acts, _, _ = encoder(acts)
    return mid_acts





def get_token_kurtosis_for_decoder(model: HookedTransformer, layer: int, decoder: torch.Tensor):
    '''Return excess kurtosis over all decoder features' cosine sims with the unembed (higher is better)'''
    W_out = model.W_out[layer]
    resid_dirs = torch.nn.functional.normalize(decoder @ W_out, dim=-1)
    unembed = torch.nn.functional.normalize(model.unembed.W_U, dim=0)
    cosine_sims = einops.einsum(resid_dirs, unembed, 'd_hidden d_model, d_model d_vocab -> d_hidden d_vocab')
    
    mean = einops.repeat(cosine_sims.mean(dim=-1), f'd_hidden -> d_hidden {cosine_sims.shape[1]}')
    std = einops.repeat(cosine_sims.std(dim=-1), f'd_hidden -> d_hidden {cosine_sims.shape[1]}')
    kurt = torch.mean(((cosine_sims - mean) / std) ** 4, dim=-1) - 3
    return kurt

def get_top_activating_examples_for_direction(prompts, direction, max_activations_per_prompt: Tensor, max_activation_token_indices, k=10, mode: Literal["lower", "middle", "upper", "top"]="top"):
    
    sorted_activations = max_activations_per_prompt[:, direction].sort().values
    num_non_zero_activations = sorted_activations[sorted_activations > 0].shape[0]

    max_activation = sorted_activations[-1]
    if mode=="upper":
        index = torch.argwhere(sorted_activations > ((max_activation // 3) * 2)).min()
    elif mode == "middle":
        index = torch.argwhere(sorted_activations > ((max_activation // 3))).min()
    else:
        index = torch.argwhere(sorted_activations > ((max_activation // 10))).min()
    negative_index = sorted_activations.shape[0] - index

    activations = max_activations_per_prompt[:, direction]
    _, prompt_indices = activations.topk(num_non_zero_activations+1)
    if mode=="top":
        prompt_indices = prompt_indices[:k]
    else:
        prompt_indices = prompt_indices[negative_index:negative_index+k]
    prompt_indices = prompt_indices[:num_non_zero_activations]

    top_prompts = [prompts[i] for i in prompt_indices]
    token_indices = max_activation_token_indices[prompt_indices, direction]
    return top_prompts, token_indices

def get_direction_ablation_hook(encoder, direction, hook_pos=None):
    def subtract_direction_hook(value, hook):
        
        x_cent = value[0, :] - encoder.b_dec
        acts = F.relu(x_cent @ encoder.W_enc[:, direction] + encoder.b_enc[direction])
        direction_impact_on_reconstruction = einops.einsum(acts, encoder.W_dec[direction, :], "pos, d_mlp -> pos d_mlp") # + encoder.b_dec ???
        if hook_pos is not None:
            value[:, hook_pos, :] -= direction_impact_on_reconstruction[hook_pos]
        else:
            value[:, :] -= direction_impact_on_reconstruction
        return value
    return subtract_direction_hook

def evaluate_direction_ablation_single_prompt(prompt: str, encoder: AutoEncoder, model: HookedTransformer, direction: int, cfg: AutoEncoderConfig, pos: None | int = None) -> float:
    encoder_hook_point = f"blocks.{cfg.layer}.{cfg.act_name}"
    if pos is not None:
        original_loss = model(prompt, return_type="loss", loss_per_token=True)[0, pos]
    else:
        original_loss = model(prompt, return_type="loss")
    
    with model.hooks(fwd_hooks=[(encoder_hook_point, get_direction_ablation_hook(encoder, direction, pos))]):
        if pos is not None:
            ablated_loss = model(prompt, return_type="loss", loss_per_token=True)[0, pos]
        else:
            ablated_loss = model(prompt, return_type="loss")
    return original_loss.item(), ablated_loss.item()

def eval_direction_tokens(direction, max_activations, max_activation_token_indices, prompts, model, encoder, cfg, percentage_threshold = 0.25, normalize_str_tokens=True):
    max_activation_value = max_activations[:, direction].max().item()
    threshold = max_activation_value * percentage_threshold
    num_non_zero_activations = (max_activations[:, direction]>threshold).sum()
    top_prompts, top_prompt_token_indices = get_top_activating_examples_for_direction(prompts, direction, max_activations, max_activation_token_indices, k=num_non_zero_activations, mode="top")

    activating_tokens = []
    for prompt, index in zip(top_prompts, top_prompt_token_indices.tolist()):
        str_tokens = model.to_str_tokens(prompt)
        activations = get_acts(prompt, model, encoder, cfg)
        activation = activations[index, direction].item()
        token = str_tokens[index]
        if normalize_str_tokens:
            token = token.strip().lower()
        if activation > threshold:
            activating_tokens.append(token)
        else:
            break
    token_counts = Counter(activating_tokens)
    return token_counts, threshold

def eval_direction_tokens_global(max_activations, prompts, model, encoder, cfg, percentage_threshold = 0.25):
    max_activation_value = max_activations.max(dim=0)[0]
    threshold_per_direction = max_activation_value * percentage_threshold
    token_wise_activations = torch.zeros((encoder.d_hidden, model.cfg.d_vocab), dtype=torch.int32)

    for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        tokens = model.to_tokens(prompt).flatten().cpu()
        activations = get_acts(prompt, model, encoder, cfg).cpu()
        num_tokens = activations.shape[0]
        for position in range(10, num_tokens):
            valid_directions = torch.argwhere(activations[position] > threshold_per_direction).flatten().cpu()
            if len(valid_directions) > 0:
                token_wise_activations[valid_directions, tokens[position]] += 1
    return token_wise_activations


def final_token_indices(model: HookedTransformer, tokens: torch.tensor):
    bos = model.tokenizer.bos_token_id
    mask = tokens != bos
    cumulative_mask = mask.cumsum(dim=1)
    return cumulative_mask.argmax(dim=1)

    # Select elements from each row up to the last non-bos index
    # return tokens[:, max_indices + 1 - neg_pos]    