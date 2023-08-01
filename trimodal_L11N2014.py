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

#%% 
# 2014: es, sv, de, en
# 1819: de, nl, it, (en)
languages = ["es", "sv", "de", "en"]

with open("./data/wmt_europarl.pkl", "rb") as f:
    data = pickle.load(f)

#%%
language_activations = []
n = 200
for language in tqdm(languages):
    language_activation = haystack_utils.get_mlp_activations([x["text"] for x in data[language][:200]], LAYER, model, neurons=torch.LongTensor([NEURON]), mean=False).flatten()
    language_activations.append(language_activation.cpu().numpy())

#%%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_histograms(data, names):
    fig = make_subplots()
    for arr, name in zip(data, names):
        fig.add_trace(go.Histogram(x=arr, name=name, nbinsx=500, opacity=0.5, histnorm='probability density'))
    fig.update_layout(barmode='overlay')
    fig.show()

plot_histograms(language_activations, languages)


# %%
ranges = [(-10, 0.2), (1, 5), (9, 50)]
labels = ['Inactive', 'Peak 1', 'Peak 2']
plotting_utils.color_binned_histogram(neuron_activations.cpu().numpy(),  ranges, labels, title=f'L{LAYER}N{NEURON} activations on German data')

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
prompt = german_data[0]

pythia_160m_utils.print_prompt(prompt, model, trimodal_interest_measure, LAYER, NEURON)
# %%
