# %%
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from jaxtyping import Float, Int, Bool
from torch import Tensor
from tqdm.auto import tqdm
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import plotly.express as px 
from collections import defaultdict
import matplotlib.pyplot as plt
import re
from IPython.display import display, HTML
from datasets import load_dataset
from collections import Counter
import pickle
import os
from typing import Literal

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

pio.renderers.default = "notebook_connected+notebook"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)

from haystack_utils import get_mlp_activations
from hook_utils import save_activation
import haystack_utils
import hook_utils
import plotting_utils

%reload_ext autoreload
%autoreload 2

# %%
model = HookedTransformer.from_pretrained("EleutherAI/pythia-160m",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device)

german_data = haystack_utils.load_json_data("data/german_europarl.json")[:200]

# %%
LAYER, NEURON = 5, 2525
LAYER_2, NEURON_2 = 3, 55

# %%

activations = []
labels = []
for prompt in german_data[:100]:
    tokens = model.to_tokens(prompt)
    mask = haystack_utils.get_next_token_punctuation_mask(tokens.flatten(), model, fill_last_pos=False)
    _, cache = model.run_with_cache(tokens)
    neuron_activations = cache[f'blocks.{LAYER}.mlp.hook_post'][0, :-1, NEURON]
    activations.append(neuron_activations)
    labels.append(mask)
activations = torch.cat(activations)
mask = torch.cat(labels)
#%%
haystack_utils.two_histogram(activations[mask], activations[~mask], "Next is space", "Next is continuation", histnorm="percent", title=f"L{LAYER}N{NEURON} activations")
# %%
# Snapping hooks for all position and extreme peaks
def snap_to_peak_1(value, hook):
    value[:, :, NEURON] = -0.1
    return value

def snap_to_peak_2(value, hook):
    value[:, :, NEURON] = 1.5
    return value

def get_peak_swap_hook(mask):
    def peak_swap(value, hook):
        value[:, ~mask, NEURON] = -0.1
        value[:, mask, NEURON] = 1.5
        return value
    return [(f'blocks.{LAYER}.mlp.hook_post', peak_swap)]

snap_to_peak_1_hook = [(f'blocks.{LAYER}.mlp.hook_post', snap_to_peak_1)]
snap_to_peak_2_hook = [(f'blocks.{LAYER}.mlp.hook_post', snap_to_peak_2)]

def snap_to_peak_1_2(value, hook):
    value[:, :, NEURON_2] = -0.1
    return value

def snap_to_peak_2_2(value, hook):
    value[:, :, NEURON_2] = 1.5
    return value

def get_peak_swap_hook_2(mask):
    def peak_swap(value, hook):
        value[:, ~mask, NEURON_2] = -0.1
        value[:, mask, NEURON_2] = 1.5
        return value
    return [(f'blocks.{LAYER_2}.mlp.hook_post', peak_swap)]

snap_to_peak_1_hook_2 = [(f'blocks.{LAYER_2}.mlp.hook_post', snap_to_peak_1_2)]
snap_to_peak_2_hook_2 = [(f'blocks.{LAYER_2}.mlp.hook_post', snap_to_peak_2_2)]
# %%

original_losses = []
losses_peak_1 = []
losses_peak_2 = []
losses_swap_peak = []
for prompt in tqdm(german_data[:200]):
    tokens = model.to_tokens(prompt)
    mask = haystack_utils.get_next_token_punctuation_mask(tokens.flatten(), model, fill_last_pos=True)
    swap_peak_hook = get_peak_swap_hook(mask)
    swap_peak_hook_2 = get_peak_swap_hook_2(mask)
    original_loss = model(tokens, return_type="loss")
    with model.hooks(snap_to_peak_1_hook + snap_to_peak_1_hook_2):
        loss_peak_1 = model(tokens, return_type="loss")
    with model.hooks(snap_to_peak_2_hook + snap_to_peak_2_hook_2):
        loss_peak_2 = model(tokens, return_type="loss")
    with model.hooks(swap_peak_hook + swap_peak_hook_2):
        loss_swap_peak = model(tokens, return_type="loss")
    original_losses.append(original_loss.item())
    losses_peak_1.append(loss_peak_1.item())
    losses_peak_2.append(loss_peak_2.item())
    losses_swap_peak.append(loss_swap_peak.item())

# %%
names = ["original", "peak_1", "peak_2", "swap"]
plotting_utils.plot_barplot([original_losses, losses_peak_1, losses_peak_2, losses_swap_peak], names, title=f"both neurons peak swap losses")
# %%
