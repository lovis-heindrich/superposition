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

model = HookedTransformer.from_pretrained("EleutherAI/pythia-160m",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device)

german_data = haystack_utils.load_json_data("data/german_europarl.json")[:200]
english_data = haystack_utils.load_json_data("data/english_europarl.json")[:200]

LAYER, NEURON = 8, 2994
neuron_activations = haystack_utils.get_mlp_activations(german_data, LAYER, model, neurons=torch.LongTensor([NEURON]), mean=False).flatten()

def get_next_token_punctuation_mask(tokens: torch.LongTensor) -> torch.BoolTensor:
    next_token_punctuation_mask = torch.zeros_like(tokens, dtype=torch.bool)
    token_strs = model.to_str_tokens(tokens)
    for i in range(tokens.shape[0] - 1):
        next_token_str = token_strs[i + 1]
        next_is_space = next_token_str[0] in [" ", ",", ".", ":", ";", "!", "?"]
        next_token_punctuation_mask[i] = next_is_space
    return next_token_punctuation_mask
# %%

def get_new_word_dense_probe_f1(model, german_data, layer, neurons=torch.LongTensor([NEURON])) -> tuple[np.ndarray, np.ndarray]:
    ''' Compute F1 of predicting word end from activation and plot performance of each peak'''
    def get_new_word_labeled_activations():
        '''Get activations and labels for predicting word end from activation'''
        hook_name = f'blocks.{layer}.mlp.hook_post'
        activations = []
        labels = []
        for prompt in german_data:
            tokens = model.to_tokens(prompt)[0]
            prompt_activations = []
            with model.hooks([(hook_name, save_activation)]):
                model(tokens)
            prompt_activations = model.hook_dict[hook_name].ctx['activation'][0, :-1, neurons].cpu().numpy()
            activations.append(prompt_activations)
            
            
            prompt_labels = []
            for i in range(tokens.shape[0] - 1):
                next_token_str = model.to_single_str_token(tokens[i + 1].item())
                next_is_space = next_token_str[0] in [" "] # [" ", ",", ".", ":", ";", "!", "?"]
                prompt_labels.append(next_is_space)
            labels.extend(prompt_labels)

        activations = np.concatenate(activations, axis=0)
        labels = np.array(labels)
        assert activations.shape[0] == labels.shape[0]

        return activations, labels

    x, y = get_new_word_labeled_activations()
    from sklearn import preprocessing

    # z-scoring can help with convergence
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)
    
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(x[:20000], y[:20000])
    preds = lr_model.predict(x[20000:])
    score = f1_score(y[20000:], preds)
    print(score)

# %% 
get_new_word_dense_probe_f1(model, german_data, LAYER, torch.LongTensor([NEURON]))
# 88.82 f1
# get_new_word_dense_probe_f1(model, german_data, LAYER, torch.LongTensor([neuron for neuron in range(model.cfg.d_mlp) if neuron != NEURON]))
# %%
# 88.83 f1
# get_new_word_dense_probe_f1(model, german_data, LAYER, torch.LongTensor([neuron for neuron in range(model.cfg.d_mlp)]))


# %%
def plot_peak_new_word_accuracies(model, german_data, layer, neuron):
    '''Plot % of each activation peak that corresponds to word end tokens'''
    def trimodal_interest_measure(activations):
        diffs = torch.zeros_like(activations) + 3
        diffs[(activations < 0.2)] = 0
        diffs[(activations > 0.8) & (activations < 4.1)] = 1
        diffs[(activations > 4.8)] = 2
        return diffs

    hook_name = f'blocks.{layer}.mlp.hook_post'
    data = []
    for prompt in german_data:
        tokens = model.to_tokens(prompt)[0]
        
        with model.hooks([(hook_name, save_activation)]):
            model(prompt)
        pos_wise_diff = trimodal_interest_measure(model.hook_dict[hook_name].ctx['activation'][0, :, neuron])
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
plot_peak_new_word_accuracies(model, german_data, LAYER, NEURON)
