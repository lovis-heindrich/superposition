import random
import argparse
import pickle
import os
import gzip
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import torch
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display, HTML
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import plotly.express as px
from nltk import ngrams

from neel_plotly import *
import haystack_utils
import probing_utils


def eval_prompts(prompts, model, pos=-1):
    '''Mean loss at position in prompts'''
    loss = model(prompts, return_type="loss", loss_per_token=True)[:, pos].mean().item()
    return loss


def deactivate_neurons_hook(value, hook):
    value[:, :, NEURON] = MEAN_ACTIVATION_INACTIVE
    return value
deactivate_neurons_fwd_hooks=[(f'blocks.{LAYER}.mlp.hook_post', deactivate_neurons_hook)]


def get_ngram_losses(
    model: HookedTransformer,
    checkpoint: int,
    ngrams: list[str],
    common_tokens: list[str],
) -> pd.DataFrame:
    data = []
    for ngram in ngrams:
        prompts = haystack_utils.generate_random_prompts(
            ngram, model, common_tokens, 100, 20
        )
        loss = eval_prompts(prompts, model)
        with model.hooks(deactivate_neurons_fwd_hooks):
            ablated_loss = eval_prompts(prompts, model)
        data.append([loss, ablated_loss, ablated_loss - loss, checkpoint, ngram])

    df = pd.DataFrame(
        data,
        columns=["OriginalLoss", "AblatedLoss", "LossIncrease", "Checkpoint", "Ngram"],
    )
    return df


def get_common_ngrams(
    model: HookedTransformer, prompts: list[str], n: int, top_k=100
) -> list[str]:
    """
    n: n-gram length
    top_k: number of n-grams to return

    Returns: List of most common n-grams in prompts sorted by frequency
    """
    all_ngrams = []
    for prompt in tqdm(prompts):
        str_tokens = model.to_str_tokens(prompt)
        all_ngrams.extend(ngrams(str_tokens, n))
    # Filter n-grams which contain punctuation
    all_ngrams = [
        x
        for x in all_ngrams
        if all(
            [
                y.strip() not in ["\n", "-", "(", ")", ".", ",", ";", "!", "?", ""]
                for y in x
            ]
        )
    ]
    return Counter(all_ngrams).most_common(top_k)

    
all_ignore, _ = haystack_utils.get_weird_tokens(model, plot_norms=False)    

common_tokens = haystack_utils.get_common_tokens(
    german_data, model, all_ignore, k=100
)
top_trigrams = get_common_ngrams(model, lang_data["de"], 3, 200)
random_trigram_indices = np.random.choice(
    range(len(top_trigrams)), 20, replace=False
)
random_trigrams = ["".join(top_trigrams[i][0]) for i in random_trigram_indices]

ngram_loss_dfs = []
with tqdm(total=num_checkpoints * n_layers) as pbar:
    for checkpoint in range(num_checkpoints):
        model = get_model(checkpoint)
        for layer in range(n_layers):
            ngram_loss_dfs.append(
                get_ngram_losses(model, checkpoint, random_trigrams, common_tokens)
            )