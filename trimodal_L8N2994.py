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
data = []
for prompt in german_data:
    tokens = model.to_tokens(prompt)[0]
    _, cache = model.run_with_cache(prompt)
    hook_name = f'blocks.{LAYER}.mlp.hook_post'
    with model.hooks([(hook_name, save_activation)]):
        model(prompt)
    pos_wise_diff = trimodal_interest_measure(model.hook_dict[hook_name].ctx['activation'][0, :, NEURON])
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
def get_next_token_punctuation_mask(tokens: torch.LongTensor) -> torch.BoolTensor:
    next_token_punctuation_mask = torch.zeros_like(tokens, dtype=torch.bool)
    token_strs = model.to_str_tokens(tokens)
    for i in range(tokens.shape[0] - 1):
        next_token_str = token_strs[i + 1]
        next_is_space = next_token_str[0] in [" ", ",", ".", ":", ";", "!", "?"]
        next_token_punctuation_mask[i] = next_is_space
    return next_token_punctuation_mask

def get_ablate_at_mask_hooks(mask: torch.BoolTensor, act_value=3.2) -> list[tuple[str, callable]]:
    def ablate_neuron_hook(value, hook):
        value[:, mask, NEURON] = act_value
    hook_name = f'blocks.{LAYER}.mlp.hook_post'
    return [(hook_name, ablate_neuron_hook)]

original_losses = []
ablated_losses = []
ablated_final_token_losses = []
ablated_other_token_losses = []
original_final_token_losses = []
original_other_token_losses = []
for prompt in tqdm(german_data):
    tokens = model.to_tokens(prompt)[0]
    mask = get_next_token_punctuation_mask(tokens)
    with model.hooks(get_ablate_at_mask_hooks(mask)):
        ablated_loss = model(prompt, return_type='loss', loss_per_token=True).flatten()

        ablated_final_token_losses.extend(ablated_loss[mask[:-1]].tolist()[5:])
        ablated_other_token_losses.extend(ablated_loss[~mask[:-1]].tolist()[5:])
    
    original_loss = model(prompt, return_type='loss', loss_per_token=True).flatten()

    original_final_token_losses.extend(original_loss[mask[:-1]].tolist()[5:])
    original_other_token_losses.extend(original_loss[~mask[:-1]].tolist()[5:])
    original_losses.extend(original_loss.tolist()[5:])

print(
    np.mean(original_losses),
    np.mean(original_other_token_losses),
    np.mean(original_final_token_losses),
    np.mean(ablated_final_token_losses))
# %%

original_losses = []
ablated_losses = []
ablated_final_token_losses = []
ablated_other_token_losses = []
original_final_token_losses = []
original_other_token_losses = []
for prompt in tqdm(german_data):
    tokens = model.to_tokens(prompt)[0]
    mask = get_next_token_punctuation_mask(tokens)

    original_loss = model(prompt, return_type='loss', loss_per_token=True).flatten()
    
    original_final_token_losses.extend(original_loss[mask[:-1]].tolist()[5:])
    original_other_token_losses.extend(original_loss[~mask[:-1]].tolist()[5:])
    original_losses.extend(original_loss.tolist()[5:])

    with model.hooks(get_ablate_at_mask_hooks(mask, act_value=neuron_activations.mean())):
        ablated_loss = model(prompt, return_type='loss', loss_per_token=True).flatten()

    ablated_final_token_losses.extend(ablated_loss[mask[:-1]].tolist()[5:])
    ablated_other_token_losses.extend(ablated_loss[~mask[:-1]].tolist()[5:])
    ablated_losses.extend(ablated_loss.tolist()[5:])


print(
    np.mean(original_losses),
    np.mean(ablated_losses),
    "\n original and ablated other token:",
    np.mean(original_other_token_losses),
    np.mean(ablated_other_token_losses),
    "\n original and ablated final token:",
    np.mean(original_final_token_losses),
    np.mean(ablated_final_token_losses))

# %%
def interest_measure(original_loss, ablated_loss):
    """Per-token measure, mixture of overall loss increase and loss increase from ablating MLP11"""
    loss_diff = (ablated_loss - original_loss) # Loss increase from context neuron
    loss_diff[original_loss > 6] = 0
    loss_diff[original_loss > ablated_loss] = 0
    return loss_diff

# %%
def print_prompt(prompt: str):
    """Red/blue scale showing the interest measure for each token"""
    tokens = model.to_tokens(prompt)[0]
    str_token_prompt = model.to_str_tokens(tokens)
    mask = get_next_token_punctuation_mask(tokens)
    with model.hooks(get_ablate_at_mask_hooks(mask, 5.5)):
        ablated_loss = model(prompt, return_type='loss', loss_per_token=True).flatten()
    original_loss = model(prompt, return_type='loss', loss_per_token=True).flatten()
    
    pos_wise_diff = interest_measure(original_loss, ablated_loss).flatten().cpu().tolist()

    loss_list = [loss.flatten().cpu().tolist() for loss in [original_loss, ablated_loss]]
    loss_names = ["original_loss", "ablated_loss"]
    haystack_utils.clean_print_strings_as_html(str_token_prompt[1:], pos_wise_diff, max_value=4, additional_measures=loss_list, additional_measure_names=loss_names)

for prompt in german_data[:5]:
    print_prompt(prompt)

# %%
def print_prompt(prompt: str):
    """Red/blue scale showing the interest measure for each token"""
    tokens = model.to_tokens(prompt)[0]
    str_token_prompt = model.to_str_tokens(tokens)
    mask = get_next_token_punctuation_mask(tokens)
    with model.hooks(get_ablate_at_mask_hooks(~mask, 5.5)):
        ablated_loss = model(prompt, return_type='loss', loss_per_token=True).flatten()
    original_loss = model(prompt, return_type='loss', loss_per_token=True).flatten()
    
    pos_wise_diff = interest_measure(original_loss, ablated_loss).flatten().cpu().tolist()

    loss_list = [loss.flatten().cpu().tolist() for loss in [original_loss, ablated_loss]]
    loss_names = ["original_loss", "ablated_loss"]
    haystack_utils.clean_print_strings_as_html(str_token_prompt[1:], pos_wise_diff, max_value=4, additional_measures=loss_list, additional_measure_names=loss_names)

for prompt in german_data[:5]:
    print_prompt(prompt)

# %%
def print_prompt(prompt: str, fwd_hooks):
    """Red/blue scale showing the interest measure for each token"""
    tokens = model.to_tokens(prompt)[0]
    str_token_prompt = model.to_str_tokens(tokens)
    mask = get_next_token_punctuation_mask(tokens)
    with model.hooks(fwd_hooks):
        ablated_loss = model(prompt, return_type='loss', loss_per_token=True).flatten()
    original_loss = model(prompt, return_type='loss', loss_per_token=True).flatten()
    
    pos_wise_diff = interest_measure(original_loss, ablated_loss).flatten().cpu().tolist()

    loss_list = [loss.flatten().cpu().tolist() for loss in [original_loss, ablated_loss]]
    loss_names = ["original_loss", "ablated_loss"]
    haystack_utils.clean_print_strings_as_html(str_token_prompt[1:], pos_wise_diff, max_value=4, additional_measures=loss_list, additional_measure_names=loss_names)

for prompt in german_data[:5]:
    print_prompt(prompt, [hook_utils.get_ablate_neuron_hook(LAYER, NEURON, 5.5)])

# %%
def trimodal_hook_1(value, hook):
    first_mode, second_mode, third_mode = 0, 3.5, 5.5
    neuron_act = value[:, :, NEURON]
    diffs = torch.stack([neuron_act - first_mode, neuron_act - second_mode, neuron_act - third_mode]).cuda()
    diffs = torch.abs(diffs)
    _, min_indices = torch.min(diffs, dim=0)

    value[:, :, NEURON] = torch.where(min_indices == 0, neuron_act, 
                                      torch.where(min_indices == 1, third_mode, third_mode))
    return value

for prompt in german_data[:5]:
    print_prompt(prompt, [(f'blocks.{LAYER}.mlp.hook_post', trimodal_hook_1)])
# %%
def trimodal_hook_2(value, hook):
    first_mode, second_mode, third_mode = 0, 3.5, 5.5
    neuron_act = value[:, :, NEURON]
    diffs = torch.stack([neuron_act - first_mode, neuron_act - second_mode, neuron_act - third_mode]).cuda()
    diffs = torch.abs(diffs)
    _, min_indices = torch.min(diffs, dim=0)

    value[:, :, NEURON] = torch.where(min_indices == 0, neuron_act, 
                                      torch.where(min_indices == 1, second_mode, second_mode))
    return value

for prompt in german_data[:5]:
    print_prompt(prompt, [(f'blocks.{LAYER}.mlp.hook_post', trimodal_hook_2)])
# %%
