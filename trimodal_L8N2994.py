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
# px.histogram(neuron_activations.cpu().numpy())
ranges, labels = [(-10, 0.2), (0.8, 4.1), (4.8, 10)], ['Inactive', 'Peak 1', 'Peak 2']
plotting_utils.color_binned_histogram(neuron_activations.cpu().numpy(), ranges, labels, title=f'L{LAYER}N{NEURON} activations on German data')
# %%
def compute_new_word_f1(model, german_data, layer, neuron):
    ''' Compute F1 of predicting word end from activation and plot performance of each peak'''
    def get_new_word_labeled_activations():
        '''Get activations and labels for predicting word end from activation'''
        activations = []
        labels = []
        for prompt in german_data:
            prompt_activations = []
            prompt_labels = []
            tokens = model.to_tokens(prompt)[0]
            hook_name = f'blocks.{layer}.mlp.hook_post'
            with model.hooks([(hook_name, save_activation)]):
                model(prompt)
            prompt_activations = model.hook_dict[hook_name].ctx['activation'][0, :-1, neuron].flatten().tolist()
            for i in range(tokens.shape[0] - 1):
                next_token_str = model.to_single_str_token(tokens[i + 1].item())
                next_is_space = next_token_str[0] in [" ", ",", ".", ":", ";", "!", "?"]
                prompt_labels.append(next_is_space)
            activations.extend(prompt_activations)
            labels.extend(prompt_labels)
            assert len(prompt_activations) == len(prompt_labels)
        return activations, labels

    activations, labels = get_new_word_labeled_activations()
    x = np.array(activations).reshape(-1, 1)
    y = np.array(labels)
    lr_model = LogisticRegression()
    lr_model.fit(x[:8000], y[:8000])
    preds = lr_model.predict(x[8000:])
    score = f1_score(y[8000:], preds)
    print(score)

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
compute_new_word_f1(model, german_data, LAYER, NEURON)
plot_peak_new_word_accuracies(model, german_data, LAYER, NEURON)
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

# %%
def print_conditional_ablation_losses(model=model, german_data=german_data, neuron_activations=neuron_activations):
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

print_conditional_ablation_losses()
# %%
def interest_measure(original_loss: torch.FloatTensor, ablated_loss: torch.FloatTensor):
    """Per-token measure, mixture of overall loss increase and loss increase from ablating MLP11"""
    loss_diff = (ablated_loss - original_loss) # Loss increase from context neuron
    loss_diff[original_loss > 6] = 0
    loss_diff[original_loss > ablated_loss] = 0
    return loss_diff

def print_prompt(prompt: str, fwd_hooks: list[tuple[str, callable]]):
    """Red/blue scale showing the interest measure for each token"""
    tokens = model.to_tokens(prompt)[0]
    str_token_prompt = model.to_str_tokens(tokens)
    with model.hooks(fwd_hooks):
        ablated_loss = model(prompt, return_type='loss', loss_per_token=True).flatten()
    original_loss = model(prompt, return_type='loss', loss_per_token=True).flatten()
    
    pos_wise_diff = interest_measure(original_loss, ablated_loss).flatten().cpu().tolist()

    loss_list = [loss.flatten().cpu().tolist() for loss in [original_loss, ablated_loss]]
    loss_names = ["original_loss", "ablated_loss"]
    haystack_utils.clean_print_strings_as_html(str_token_prompt[1:], pos_wise_diff, max_value=4, additional_measures=loss_list, additional_measure_names=loss_names)

def snap_to_closest_peak(value, hook):
    '''Doesn't snap disabled and ambiguous activations'''
    neuron_act = value[:, :, NEURON]
    value[:, :, NEURON][(neuron_act > 0.8) & (neuron_act < 4.1)] = 3.5
    value[:, :, NEURON][(neuron_act > 4.8)] = 5.5
    return value

def snap_to_peak_1(value, hook):
    '''Doesn't snap disabled and ambiguous activations'''
    neuron_act = value[:, :, NEURON]
    value[:, :, NEURON][(neuron_act > 0.8) & (neuron_act < 4.1)] = 3.5
    value[:, :, NEURON][(neuron_act > 4.8)] = 3.5
    return value

def snap_to_peak_2(value, hook):
    '''Doesn't snap disabled and ambiguous activations'''
    neuron_act = value[:, :, NEURON]
    value[:, :, NEURON][(neuron_act > 0.8) & (neuron_act < 4.1)] = 5.5
    value[:, :, NEURON][(neuron_act > 4.8)] = 5.5
    return value
# %%
for prompt in german_data[:5]:
    tokens = model.to_tokens(prompt)[0]
    mask = get_next_token_punctuation_mask(tokens)
    print_prompt(prompt, get_ablate_at_mask_hooks(mask, 5.5))
# %%
for prompt in german_data[:5]:
    tokens = model.to_tokens(prompt)[0]
    mask = get_next_token_punctuation_mask(tokens)
    print_prompt(prompt, get_ablate_at_mask_hooks(~mask, 5.5))
# %%
for prompt in german_data[:5]:
    print_prompt(prompt, [hook_utils.get_ablate_neuron_hook(LAYER, NEURON, 5.5)])
# %%
for prompt in german_data[:5]:
    print_prompt(prompt, [(f'blocks.{LAYER}.mlp.hook_post', snap_to_peak_2)])
# %%
for prompt in german_data[:5]:
    print_prompt(prompt, [(f'blocks.{LAYER}.mlp.hook_post', snap_to_peak_1)])
# %%
german_data = haystack_utils.load_json_data("data/german_europarl.json")
# %%
def get_peak_losses(model: HookedTransformer, data: list[str]) -> pd.DataFrame:
    results = [] # cols: loss snapping_mode mask

    for prompt in tqdm(data):
        tokens = model.to_tokens(prompt)[0]
        mask = get_next_token_punctuation_mask(tokens)[:-1].cpu()

        with model.hooks([(f'blocks.{LAYER}.mlp.hook_post', snap_to_closest_peak)]):
            loss = model(prompt, return_type='loss', loss_per_token=True).flatten().cpu()
        results.append([loss.mean().item(), "closest_peak", "all"])
        results.append([loss[mask].mean().item(), "closest_peak", "final_token"])
        results.append([loss[~mask].mean().item(), "closest_peak", "other_token"])
    
        with model.hooks([(f'blocks.{LAYER}.mlp.hook_post', snap_to_peak_1)]):
            loss = model(prompt, return_type='loss', loss_per_token=True).flatten().cpu()
            results.append([loss.mean().item(), "peak_1", "all"])
        results.append([loss[mask].mean().item(), "peak_1", "final_token"])
        results.append([loss[~mask].mean().item(), "peak_1", "other_token"])

        with model.hooks([(f'blocks.{LAYER}.mlp.hook_post', snap_to_peak_2)]):
            loss = model(prompt, return_type='loss', loss_per_token=True).flatten().cpu()
            results.append([loss.mean().item(), "peak_2", "all"])
        results.append([loss[mask].mean().item(), "peak_2", "final_token"])
        results.append([loss[~mask].mean().item(), "peak_2", "other_token"])

        loss = model(prompt, return_type='loss', loss_per_token=True).flatten()
        results.append([loss.mean().item(), "none", "all"])
        results.append([loss[mask].mean().item(), "none", "final_token"])
        results.append([loss[~mask].mean().item(), "none", "other_token"])
    
    return pd.DataFrame(results, columns=["Loss", "Snapping Mode", "Mask"])

full_losses = get_peak_losses(model, german_data)
# %%
full_losses.groupby(["Mask", "Snapping Mode"]).mean()
df = full_losses

# %%
def plot_loss_diffs(df):
    df_diff = df.copy()
    for mask in ["all", "final_token", "other_token"]:
        for item in ["peak_1", "peak_2", "closest_peak"]:
            loss_item = df_diff.loc[(df_diff['Snapping Mode']==item) & (df_diff['Mask']==mask), 'Loss'].reset_index(drop=True)
            loss_none = df_diff.loc[(df_diff['Mask']==mask) & (df_diff['Snapping Mode']=='none'), 'Loss'].reset_index(drop=True)
            
            df_diff.loc[(df_diff['Snapping Mode']==item) & (df_diff['Mask']==mask), 'Loss'] = loss_item - loss_none
    df_diff = df_diff[~(df_diff['Snapping Mode']=='none')]
    df_diff_avg = df_diff.groupby(['Mask', 'Snapping Mode'])['Loss'].mean().reset_index()
    df_diff_sem = df_diff.groupby(['Mask', 'Snapping Mode'])['Loss'].sem().reset_index()
    df_diff_sem['Loss'] = df_diff_sem['Loss'] * 1.96
    
    px.bar(df_diff_avg, x="Mask", y="Loss", color="Snapping Mode", barmode="group", 
        hover_data=df_diff_avg.columns, width=800, error_y=df_diff_sem['Loss'])

plot_loss_diffs(df)
# %%
def get_tokenwise_high_loss_diffs(prompt: str, model: HookedTransformer):
    results = []

    with model.hooks([(f'blocks.{LAYER}.mlp.hook_post', snap_to_closest_peak)]):
        loss = model(prompt, return_type='loss', loss_per_token=True).flatten().cpu()
        results.append(loss)
    with model.hooks([(f'blocks.{LAYER}.mlp.hook_post', snap_to_peak_1)]):
        loss = model(prompt, return_type='loss', loss_per_token=True).flatten().cpu()
        results.append(loss)
    with model.hooks([(f'blocks.{LAYER}.mlp.hook_post', snap_to_peak_2)]):
        loss = model(prompt, return_type='loss', loss_per_token=True).flatten().cpu()
        results.append(loss)
    loss = model(prompt, return_type='loss', loss_per_token=True).flatten().cpu()
    results.append(loss)
    
    return results
# %%
def print_high_loss_prompts(model, german_data):
    german_losses = []
    for prompt in tqdm(german_data[:200]):
        closest_loss, peak_1_loss, peak_2_loss, original_loss = get_tokenwise_high_loss_diffs(prompt, model)
        diff = (peak_1_loss - peak_2_loss).abs().max()
        german_losses.append(diff)

    index = [i for i in range(len(german_losses))]
    sorted_measure = list(zip(index, german_losses))
    sorted_measure.sort(key=lambda x: x[1], reverse=True)

    for i, measure in sorted_measure[:10]:
        prompt = german_data[i]
        print(measure)
        tokens = model.to_tokens(prompt)[0]
        str_token_prompt = model.to_str_tokens(tokens)
        closest_loss, peak_1_loss, peak_2_loss, original_loss = get_tokenwise_high_loss_diffs(prompt, model)
        diff = (peak_1_loss - peak_2_loss).abs().flatten().cpu().tolist()
        haystack_utils.clean_print_strings_as_html(str_token_prompt[1:], diff, max_value=3.5, 
                                                additional_measures=[closest_loss, peak_1_loss, peak_2_loss, original_loss], 
                                                additional_measure_names=["closest_loss", "peak_1_loss", "peak_2_loss", "original_loss"])
# %%
print_high_loss_prompts(model, german_data)
# %%