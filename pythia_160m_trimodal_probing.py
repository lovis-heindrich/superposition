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
from functools import partial

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
from sklearn import preprocessing

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

def get_new_word_labels(model: HookedTransformer, tokens: torch.Tensor) -> list[bool]:
    prompt_labels = []
    for i in range(tokens.shape[0] - 1):
        next_token_str = model.to_single_str_token(tokens[i + 1].item())
        next_is_space = next_token_str[0] in [" "] # [" ", ",", ".", ":", ";", "!", "?"]
        prompt_labels.append(next_is_space)
    return prompt_labels

def get_new_word_labels_and_activations(
    model: HookedTransformer, 
    german_data: list[str], 
    activation_hook_name: str,
    activation_slice=np.s_[0, :-1, NEURON:NEURON+1]
) -> tuple[np.ndarray, np.ndarray]:
    '''Get activations and labels for predicting word end from activation'''
    activations = []
    labels = []
    for prompt in german_data:
        tokens = model.to_tokens(prompt)[0]

        with model.hooks([(activation_hook_name, save_activation)]):
                model(tokens)
        prompt_activations = model.hook_dict[activation_hook_name].ctx['activation']
        prompt_activations = prompt_activations[activation_slice].cpu().numpy()
        activations.append(prompt_activations)

        prompt_labels = get_new_word_labels(model, tokens)
        labels.extend(prompt_labels)

    activations = np.concatenate(activations, axis=0)
    labels = np.array(labels)

    assert activations.shape[0] == labels.shape[0]
    return activations, labels

def get_new_word_dense_probe_f1(x: np.ndarray, y: np.ndarray) -> float:
    # z-scoring can help with convergence
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)
    
    lr_model = LogisticRegression(max_iter=1200)
    lr_model.fit(x[:20000], y[:20000])
    preds = lr_model.predict(x[20000:])
    score = f1_score(y[20000:], preds)
    return score

# %%
hook_name = f'blocks.{LAYER}.mlp.hook_post'
x, y = get_new_word_labels_and_activations(model, german_data, hook_name)
score = get_new_word_dense_probe_f1(x, y)
print(score)
# %%
activation_slice = np.s_[0, :-1, [neuron for neuron in range(model.cfg.d_mlp) if neuron != NEURON]]
x, y = get_new_word_labels_and_activations(model, german_data, hook_name, activation_slice)
score = get_new_word_dense_probe_f1(x, y)
print(score) # 88.82 f1

# %% 
f1s = []
activation_slice = np.s_[0, :-1, :]
for layer in range(11):
    hook_name = f'blocks.{layer}.hook_mlp_out'
    x, y = get_new_word_labels_and_activations(model, german_data, hook_name, activation_slice)
    f1s.append(get_new_word_dense_probe_f1(x, y))

haystack_utils.line(f1s, title="F1 Score at MLP out by layer", xlabel="Layer", ylabel="F1 Score")
# %%
# Dense probe performance on each residual and MLP out
# We can do each MLP out using the current method
# Need a new method for the dimensions of the residual
# %%
def get_new_word_labels_and_resid_activations(
    model: HookedTransformer, 
    german_data: list[str], 
) -> tuple[defaultdict, np.ndarray]:
    '''Get activations and labels for predicting word end from activation'''
    activations = defaultdict(partial(np.ndarray, (0, model.cfg.d_model)))
    labels = []
    for prompt in german_data:
        tokens = model.to_tokens(prompt)[0]
        _, cache = model.run_with_cache(tokens)
        resid = cache.decompose_resid(apply_ln=False)[:, 0, :-1, :]
        for i in range(resid.shape[0]):
            activations[i] = np.concatenate((activations[i], resid[i].cpu().numpy()), axis=0)

        prompt_labels = get_new_word_labels(model, tokens)
        labels.extend(prompt_labels)

    labels = np.array(labels)
    return activations, labels

activations_dict, labels = get_new_word_labels_and_resid_activations(model, german_data)

f1s = []
for component in activations_dict.values():
    f1s.append(get_new_word_dense_probe_f1(component, labels))

# %% 

_, cache = model.run_with_cache(german_data[0])
_, component_labels = cache.decompose_resid(apply_ln=False, return_labels=True)
haystack_utils.line(f1s, xticks=component_labels, title="F1 Score at residual stream by layer", xlabel="Layer", ylabel="F1 Score")
# %%
l8_n2994_input = model.W_in[LAYER, :, NEURON].cpu().numpy()

projections = []
for component in activations_dict.values():
    projection = np.dot(component, l8_n2994_input)[:, np.newaxis]
    print(projection.shape, labels.shape)
    projections.append(get_new_word_dense_probe_f1(projection, labels))
haystack_utils.line(projections, xticks=labels, title="F1 Score at residual stream by layer", xlabel="Layer", ylabel="F1 Score")

# %%
