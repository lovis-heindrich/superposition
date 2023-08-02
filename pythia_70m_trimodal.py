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
