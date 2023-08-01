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
import haystack_utils
import hook_utils
import plotting_utils
import pythia_160m_utils

%reload_ext autoreload
%autoreload 2
# %%
model = HookedTransformer.from_pretrained("EleutherAI/pythia-160m",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device)


german_data = haystack_utils.load_json_data("data/german_europarl.json")[:200]
english_data = haystack_utils.load_json_data("data/english_europarl.json")[:200]
# %%

LAYER, NEURON = 11, 2014
neuron_activations = haystack_utils.get_mlp_activations(german_data, LAYER, model, neurons=torch.LongTensor([NEURON]), mean=False).flatten()
# %%
px.histogram(neuron_activations.cpu().numpy())

# %%
ranges = [(-10, 0.2), (1, 5), (9, 50)]
labels = ['Inactive', 'Peak 1', 'Peak 2']
plotting_utils.color_binned_histogram(neuron_activations.cpu().numpy(),  ranges, labels, title=f'L{LAYER}N{NEURON} activations on German data')


# %%

# %% 
def trimodal_interest_measure(activations):
    diffs = torch.zeros_like(activations) + 3
    diffs[(activations < 0.2)] = 0
    diffs[(activations > 1) & (activations < 5)] = 1
    diffs[(activations > 9)] = 2
    return diffs

# %% 
print(trimodal_interest_measure(neuron_activations[:10]))
# %% 
# prompt = german_data[0]
for prompt in german_data[:10]:
    pythia_160m_utils.print_prompt(prompt, model, trimodal_interest_measure, LAYER, NEURON)
# %%
data = []
counter = Counter()
for prompt in german_data:
    tokens = model.to_tokens(prompt)[0]
    _, cache = model.run_with_cache(prompt)
    pos_wise_diff = trimodal_interest_measure(cache['post', LAYER][0, :, NEURON])
    for i in range(tokens.shape[0] - 1):
        next_token_str = model.to_single_str_token(tokens[i + 1].item())
        next_is_space = next_token_str[0] in [" ", ",", ".", ":", ";", "!", "?"]
        if pos_wise_diff[i] == 1:
            data.append(["Peak 1", next_is_space])
        elif pos_wise_diff[i] == 2:
            data.append(["Peak 2", next_is_space])
        elif pos_wise_diff[i] == 0:
            data.append(["Inactive", next_is_space])
        else:
            assert pos_wise_diff[i] == 3
            data.append(["Unclear Peak", next_is_space])

df = pd.DataFrame(data, columns=["Peak", "BeforeNewWord"])

fig = px.histogram(df, x="Peak", color="BeforeNewWord", barmode="group", hover_data=df.columns, width=800)
fig.update_layout(
    title_text='Peak data, check if next token starts with [" ", ",", ".", ":", ";", "!", "?"]', # title of plot
    xaxis_title_text='', # xaxis label
    yaxis_title_text='Number of tokens', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
)
fig.show()
# %%

# %%
