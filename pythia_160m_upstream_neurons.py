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
def get_neuron_peaks(layer, neuron):
    activations = []
    labels = []
    for prompt in tqdm(german_data[:100]):
        tokens = model.to_tokens(prompt)
        mask = haystack_utils.get_next_token_punctuation_mask(tokens.flatten(), model, fill_last_pos=False)
        _, cache = model.run_with_cache(tokens)
        neuron_activations = cache[f'blocks.{layer}.mlp.hook_post'][0, :-1, neuron]
        activations.append(neuron_activations)
        labels.append(mask)
    activations = torch.cat(activations)
    mask = torch.cat(labels)
    #haystack_utils.two_histogram(activations[mask], activations[~mask], "Next is space", "Next is continuation", histnorm="percent", title=f"L{layer}N{neuron} activations")

    is_space_activation = activations[mask].mean()
    is_continuation_activation = activations[~mask].mean()
    return is_space_activation, is_continuation_activation

def get_snap_hooks(layer, neuron, is_space_activation, is_continuation_activation):
    def snap_to_space(value, hook):
        value[:, :, neuron] = is_space_activation
        return value

    def snap_to_continuation(value, hook):
        value[:, :, neuron] = is_continuation_activation
        return value
    
    def get_peak_swap_hook(mask):
        def peak_swap(value, hook):
            value[:, ~mask, neuron] = is_space_activation
            value[:, mask, neuron] = is_continuation_activation
            return value
        return [(f'blocks.{layer}.mlp.hook_post', peak_swap)]

    return [(f'blocks.{layer}.mlp.hook_post', snap_to_space)], [(f'blocks.{layer}.mlp.hook_post', snap_to_continuation)], get_peak_swap_hook

def get_swap_losses(model, space_hook, continuation_hook, swap_factory):
    original_losses = []
    losses_space = []
    losses_continuation = []
    losses_swap = []
    for prompt in tqdm(german_data[:200]):
        tokens = model.to_tokens(prompt)
        mask = haystack_utils.get_next_token_punctuation_mask(tokens.flatten(), model, fill_last_pos=True)
        if type(swap_factory) == list:
            swap_hook = []
            for hook in swap_factory:
                swap_hook += hook(mask)
        else:
            swap_hook = swap_factory(mask)
        original_loss = model(tokens, return_type="loss")
        with model.hooks(space_hook):
            loss_space = model(tokens, return_type="loss")
        with model.hooks(continuation_hook):
            loss_continuation = model(tokens, return_type="loss")
        with model.hooks(swap_hook):
            loss_swap = model(tokens, return_type="loss")
        original_losses.append(original_loss.item())
        losses_space.append(loss_space.item())
        losses_continuation.append(loss_continuation.item())
        losses_swap.append(loss_swap.item())
    return original_losses, losses_space, losses_continuation, losses_swap


# %%
df = pd.read_csv("./data/pythia_160m/space_probing_baseline.csv")
df = df.sort_values(by="f1 (next_is_space)", ascending=False)
df.head(20)
# %%
# L5 neurons 2525, 1373, 274, 827, 1095
# L3 neurons 55, 1550
# L8 neurons 1426, 2994, 1080, 1404, 2636, 667, 1311
l_3_neurons = [55, 1550]
l_5_neurons = [2525, 1373, 274, 827, 1095]
l_8_neurons = [1426, 2994, 1080, 1404, 2636, 667, 1311]

layers = df[df["f1 (next_is_space)"] > 0.8]["layer"].tolist()
neurons = df[df["f1 (next_is_space)"] > 0.8]["neuron"].tolist()
neurons = [(l, n) for l, n in zip(layers, neurons) if n != 2994]
print(len(neurons))
#neurons = [(3, n) for n in l_3_neurons] + [(5, n) for n in l_5_neurons] + [(8, n) for n in l_8_neurons]
#%%
is_space_activations = []
is_continuation_activations = []

for layer, neuron in neurons:
    is_space_activation, is_continuation_activation = get_neuron_peaks(layer, neuron)
    is_space_activations.append(is_space_activation)
    is_continuation_activations.append(is_continuation_activation)

print(is_space_activations)
print(is_continuation_activations)

# %%
space_hooks = []
continuation_hooks = []
swap_factories = []
for i, (layer, neuron) in enumerate(neurons):
    is_space_activation = is_space_activations[i]
    is_continuation_activation = is_continuation_activations[i]
    space_hook, continuation_hook, swap_factory = get_snap_hooks(layer, neuron, is_space_activation, is_continuation_activation)
    space_hooks += space_hook
    continuation_hooks += continuation_hook
    swap_factories.append(swap_factory)

# %%
original_losses, losses_space, losses_continuation, losses_swap = get_swap_losses(model, space_hooks, continuation_hooks, swap_factories)
#%%
plotting_utils.plot_barplot([original_losses, losses_space, losses_continuation, losses_swap], 
                            ["original", "snap to space", "snap to continuation", "swap"], 
                            #title=f"L8N2994 peak swap losses",
                            #title=f"All {len(neurons)} is_space neurons before layer 8 (f1>0.8) peak swap losses", 
                            title=f"Is_space neurons before layer 8 except L8N2994 peak swap losses", 
                            yaxis=dict(range=[2.2, 2.7]), confidence_interval=True, width=800)

# %%
w_in = model.W_in[layer, :, neurons].clone().cpu()
sims = torch.zeros(len(neurons), len(neurons))
pbar = tqdm(total=(len(neurons) * len(neurons)) / 2)
for i in range(len(neurons)):
    for j in range(i, len(neurons)):
        sim = torch.cosine_similarity(w_in[:, i], w_in[:, j], dim=0)
        sims[i, j] = sim
        pbar.update(1)
pbar.close()
# %%
names = [str(n) for n in neurons]
df = pd.DataFrame(sims.numpy(), columns=names, index=names)
fig = px.imshow(df, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title=f"L{layer} is_space neurons w_in similarity")
fig.show()
# %%
w_out = model.W_out[layer, neurons].clone().cpu()
sims = torch.zeros(len(neurons), len(neurons))
for i in tqdm(range(len(neurons))):
    for j in range(i, len(neurons)):
        sim = torch.cosine_similarity(w_out[i], w_out[j], dim=0)
        sims[i, j] = sim

# %%
names = [str(n) for n in neurons]
df = pd.DataFrame(sims.numpy(), columns=names, index=names)
fig = px.imshow(df, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title=f"L{layer} is_space neurons w_out similarity")
fig.show()
# %%


res_list = []
losses = []
for prompt in tqdm(german_data[:100]):
    tokens = model.to_tokens(prompt)
    loss, cache = model.run_with_cache(tokens, return_type="loss")
    res_post = cache["blocks.11.hook_resid_post"][0].cpu()
    res_list.append(res_post)
    losses.append(loss.item())
print(np.mean(losses))
# %%
res_post = torch.cat(res_list)
res_post_mean = res_post.mean(dim=0)
res_post_mean.shape
# %%
def ablate_res_post(value, hook):
    value[:, :] = res_post_mean
    return value
ablate_res_post_hook = [("blocks.11.hook_resid_post", ablate_res_post)]

ablated_losses = []
for prompt in tqdm(german_data[:100]):
    with model.hooks(ablate_res_post_hook):
        loss = model(prompt, return_type="loss")
    ablated_losses.append(loss.item())
print(np.mean(ablated_losses))
# %%


neurons = [(8, 2994)]
layer = 8
neuron = 2994
activations = []
for prompt in tqdm(german_data[:100]):
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)
    neuron_activations = cache[f'blocks.{layer}.mlp.hook_post'][0, :-1, neuron]
    activations.append(neuron_activations)
activations = torch.cat(activations)
print(activations.mean(), activations.shape)
mean_act = activations.mean().item()
# %%
def mean_ablate(value, hook):
    value[:, :, neuron] = mean_act
    return value

mean_ablate_hook = [(f'blocks.{layer}.mlp.hook_post', mean_ablate)]

losses = []
for prompt in tqdm(german_data[:200]):
    with model.hooks(mean_ablate_hook):
        loss = model(prompt, return_type="loss")
    losses.append(loss.item())
print(np.mean(losses))
# %%
