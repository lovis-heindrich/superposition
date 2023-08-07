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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

with open("./data/wmt_europarl.pkl", "rb") as f:
    data = pickle.load(f)
#%% 
# 2014: es, sv, de, en
# 1819: de, nl, it, (en)
LAYER, NEURON = 11, 2014
languages = ["es", "sv", "de", "en"]

# LAYER, NEURON = 11, 1819
# languages = ["de", "nl", "it", "en"]

# LAYER, NEURON = 8, 2994
# languages = ["de", "nl", "it", "en", "sv"]

#%%
language_activations = []
n = 200
for language in tqdm(languages):
    language_activation = haystack_utils.get_mlp_activations([x["text"] for x in data[language][:200]], LAYER, model, neurons=torch.LongTensor([NEURON]), mean=False).flatten()
    language_activations.append(language_activation.cpu().numpy())

#%%
def plot_histograms(data, names, title=""):
    fig = make_subplots()
    for arr, name in zip(data, names):
        fig.add_trace(go.Histogram(x=arr, name=name, nbinsx=500, opacity=0.5, histnorm='probability density'))
    fig.update_layout(barmode='overlay', title=title)
    fig.show()

plot_histograms(language_activations, languages, title=f'L{LAYER}N{NEURON} activations on different languages')

# %%
#mean_activation = np.mean(language_activations[languages.index("en")])
mean_activation = np.mean(language_activations[languages.index("de")])
print(mean_activation)
# %%
# print some high loss examples
mean_ablate_hook = hook_utils.get_ablate_neuron_hook(LAYER, NEURON, float(mean_activation))

prompts = [x["text"] for x in data["de"][:200]]

def get_loss_diff(prompt):
    loss = model(prompt, return_type="loss", loss_per_token=True)
    with model.hooks(fwd_hooks=[mean_ablate_hook]):
        ablated_loss = model(prompt, return_type="loss", loss_per_token=True)
    return (ablated_loss - loss).flatten().tolist()

for prompt in prompts[5:10]:
    diff = get_loss_diff(prompt)
    str_tokens = model.to_str_tokens(model.to_tokens(prompt).flatten())[1:]
    haystack_utils.clean_print_strings_as_html(str_tokens, diff)

# %%
ranges = [(-10, 0.2), (1, 5), (9, 50)]
labels = ['Inactive', 'Peak 1', 'Peak 2']
plotting_utils.color_binned_histogram(language_activations[languages.index("de")],  ranges, labels, title=f'L{LAYER}N{NEURON} activations on German data')

# %% 
def trimodal_interest_measure(activations):
    diffs = torch.zeros_like(activations) + 3
    diffs[(activations < 0.2)] = 0
    diffs[(activations > 1) & (activations < 5)] = 1
    diffs[(activations > 9)] = 2
    return diffs

# %%
neuron_activations = language_activations[languages.index("de")][:10]
print(trimodal_interest_measure(torch.tensor(neuron_activations)))
# %% 
for prompt in data["de"][:10]:
    prompt = prompt["text"]
    pythia_160m_utils.print_prompt(prompt, model, trimodal_interest_measure, LAYER, NEURON)
# %%
