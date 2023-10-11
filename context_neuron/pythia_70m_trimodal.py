# For many L3N669 context neuron activations plot whether the next token is a new word. This tests whether there's a
# correlation between "is German" and "is German new word" that could have eventually developed into the trimodal direction.
# - No visible correlation
# Investigate zero values of another context neuron with 3+ peaks, with the positive ones coding for both English and German.
# - Could there be a meaning to the zero values too?
# - Wes say it's probably a meaningless bias neuron and not a context neuron

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
import pythia_160m_utils

%reload_ext autoreload
%autoreload 2
# %%
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device)

german_data = haystack_utils.load_json_data("data/german_europarl.json")[:200]
english_data = haystack_utils.load_json_data("data/english_europarl.json")[:200]
# %%

LAYER, NEURON = 3, 669
neuron_activations = haystack_utils.get_mlp_activations(german_data, LAYER, model, neurons=torch.LongTensor([NEURON]), mean=False).flatten()
# %%

def get_next_token_punctuation_mask(tokens: torch.LongTensor) -> torch.BoolTensor:
    next_token_punctuation_mask = torch.zeros_like(tokens, dtype=torch.bool)
    token_strs = model.to_str_tokens(tokens)
    for i in range(tokens.shape[0] - 1):
        next_token_str = token_strs[i + 1]
        next_is_space = next_token_str[0] in [" ", ",", ".", ":", ";", "!", "?"]
        next_token_punctuation_mask[i] = next_is_space
    return next_token_punctuation_mask

# %%
hook_name = 'blocks.3.mlp.hook_post'

next_token = []
other_token = []
labels = []

hooks = [(hook_name, save_activation)]
for prompt in german_data:
    tokens = model.to_tokens(prompt)[0]
    next_token_punctuation_mask = get_next_token_punctuation_mask(tokens).flatten()

    with model.hooks(hooks):
        model(prompt)
        acts = model.hook_dict[hook_name].ctx['activation'][0, :, NEURON]

    next_token.append(acts[next_token_punctuation_mask].cpu())
    other_token.append(acts[next_token_punctuation_mask].cpu())

next_token = torch.cat(next_token).flatten()
other_token = torch.cat(other_token).flatten()


haystack_utils.two_histogram(next_token, other_token, x_label="Neuron activation", y_label="Count", title="Neuron activation for next token punctuation")
# %%
# Investigate 70m context neurons for weirdness
all_german_neurons = [
    (5, 1039), # en
    (5, 407), # en and german
    (5, 1516), # en german
    (5,	250), # off for both
    (3, 1204), # off for german
    # (4, 482),
    # (5, 1336),
    # (4, 326),
    # (3, 669),
    # (4, 1903),
]
for layer, neuron in all_german_neurons:
    plotting_utils.plot_neuron_acts(model, german_data, [(layer, neuron)])
    plotting_utils.plot_neuron_acts(model, english_data, [(layer, neuron)])

# %%
LAYER = 5
NEURON = 407

def trimodal_interest_measure(l5_n407_acts):
    diffs = torch.zeros_like(l5_n407_acts) + 3
    diffs[(l5_n407_acts < 0.2)] = 0
    diffs[(l5_n407_acts > 1) & (l5_n407_acts < 3)] = 1
    diffs[(l5_n407_acts > 5)] = 2
    return diffs

def color_scale(diffs):
    diffs = diffs.float()
    diffs[diffs == 0] = 0.2
    diffs[diffs == 1] = 0.4
    diffs[diffs == 2] = 0.8
    diffs[diffs == 3] = 1.0
    return diffs
# %%
for prompt in german_data[:5]:
    str_token_prompt = model.to_str_tokens(model.to_tokens(prompt))
    activations = haystack_utils.get_mlp_activations([prompt], LAYER, model, neurons=torch.LongTensor([NEURON]), mean=False, context_crop_start=0, context_crop_end=20000).flatten()
    interest = trimodal_interest_measure(activations)
    pythia_160m_utils.color_strings_by_value(str_token_prompt, color_scale(interest), additional_measures=[interest])
# %%

pile_sample = load_dataset("NeelNanda/pile-10k", split='train')
# %%
print(type(pile_sample[0]['text']))
# %%
print("blue is unclassified color")
for i in range(100, 150):
    print(i)
    prompt = pile_sample[i]['text']
    str_token_prompt = model.to_str_tokens(model.to_tokens(prompt))
    activations = haystack_utils.get_mlp_activations([prompt], LAYER, model, neurons=torch.LongTensor([NEURON]), mean=False, context_crop_start=0, context_crop_end=20000).flatten()
    interest = trimodal_interest_measure(activations)
    pythia_160m_utils.color_strings_by_value(str_token_prompt, color_scale(interest), additional_measures=[interest])
# %%
# patent office, acronyms, symbols e.g. >
# prompts of interest: 45, 148 others

french_data = haystack_utils.load_json_data("data/french_data.json")[:200]
print("blue is unclassified color")
for prompt in french_data:
    str_token_prompt = model.to_str_tokens(model.to_tokens(prompt))
    activations = haystack_utils.get_mlp_activations([prompt], LAYER, model, neurons=torch.LongTensor([NEURON]), mean=False, context_crop_start=0, context_crop_end=20000).flatten()
    interest = trimodal_interest_measure(activations)
    pythia_160m_utils.color_strings_by_value(str_token_prompt, color_scale(interest), additional_measures=[interest])

# %%
# not a context neuron, activation
# range is almost always positive
# but shoudl maybe be normalized by the variance
