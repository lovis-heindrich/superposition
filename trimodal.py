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
from hook_utils import save_activation

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

LAYER, NEURON = 8, 2994
neuron_activations = haystack_utils.get_mlp_activations(german_data, LAYER, model, neurons=torch.LongTensor([NEURON]), mean=False).flatten()
# %%
px.histogram(neuron_activations.cpu().numpy())
# %%
ranges, labels = [(-10, 0.2), (0.8, 4.1), (4.8, 10)], ['Inactive', 'Peak 1', 'Peak 2']
plotting_utils.color_binned_histogram(neuron_activations.cpu().numpy(),  ranges, labels, title=f'L{LAYER}N{NEURON} activations on German data')
# %%
activations = []
labels = []
for prompt in german_data:
    prompt_activations = []
    prompt_labels = []
    tokens = model.to_tokens(prompt)[0]
    _, cache = model.run_with_cache(prompt)
    prompt_activations = cache['post', LAYER][0, :-1, NEURON].flatten().tolist()
    for i in range(tokens.shape[0] - 1):
        next_token_str = model.to_single_str_token(tokens[i + 1].item())
        next_is_space = next_token_str[0] in [" ", ",", ".", ":", ";", "!", "?"]#next_token_str.startswith(' ') or next_token_str.startswith('.') or next_token_str.startswith(','):
        prompt_labels.append(next_is_space)
    activations.extend(prompt_activations)
    labels.extend(prompt_labels)
    assert len(prompt_activations) == len(prompt_labels)

# %%
# Compute F1 of predicting word end from activation
x = np.array(activations).reshape(-1, 1)
y = np.array(labels)

lr_model = LogisticRegression()
lr_model.fit(x[:8000], y[:8000])
preds = lr_model.predict(x[8000:])
score = f1_score(y[8000:], preds)
print(score)

# %% 
def trimodal_interest_measure(activations):
    diffs = torch.zeros_like(activations) + 3
    diffs[(activations < 0.2)] = 0
    diffs[(activations > 0.8) & (activations < 4.1)] = 1
    diffs[(activations > 4.8)] = 2
    return diffs

# %%
from typing import Callable
def get_df_for_predicates(predicate: Callable[[HookedTransformer, Tensor, int], bool]):
    data = []
    for prompt in german_data:
        tokens = model.to_tokens(prompt)[0]
        hook_name = f'blocks.{LAYER}.mlp.hook_post'
        with model.hooks([(hook_name, save_activation)]):
            model(prompt)
        pos_wise_diff = trimodal_interest_measure(model.hook_dict[hook_name].ctx['activation'][0, :, NEURON])
        for i in range(tokens.shape[0] - 1):
            result = predicate(model, tokens, i)
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

    return pd.DataFrame(data, columns=["Peak", "BeforeNewWord"])

def predicate(model, tokens, token_index):
    next_token_str = model.to_single_str_token(tokens[token_index + 1].item())
    return next_token_str[0] in [" ", ",", ".", ":", ";", "!", "?"]
    
df = get_df_for_predicates(predicate)

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
