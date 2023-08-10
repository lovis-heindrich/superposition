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
all_ignore, _ = haystack_utils.get_weird_tokens(model, plot_norms=False)
common_tokens = haystack_utils.get_common_tokens(german_data, model, all_ignore, k=100)

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

def print_prompt(prompt: str, fwd_hooks: list[tuple[str, callable]], max_value=4):
    """Red/blue scale showing the interest measure for each token"""
    tokens = model.to_tokens(prompt)[0]
    str_token_prompt = model.to_str_tokens(tokens)
    with model.hooks(fwd_hooks):
        ablated_loss = model(prompt, return_type='loss', loss_per_token=True).flatten()
    original_loss = model(prompt, return_type='loss', loss_per_token=True).flatten()
    
    pos_wise_diff = interest_measure(original_loss, ablated_loss).flatten().cpu().tolist()

    loss_list = [loss.flatten().cpu().tolist() for loss in [original_loss, ablated_loss]]
    loss_names = ["original_loss", "ablated_loss"]
    haystack_utils.clean_print_strings_as_html(str_token_prompt[1:], pos_wise_diff, max_value=max_value, additional_measures=loss_list, additional_measure_names=loss_names)

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

def get_tokenwise_high_loss_diffs(prompt: str, model: HookedTransformer):
    with model.hooks([(f'blocks.{LAYER}.mlp.hook_post', snap_to_closest_peak)]):
        snap_to_closest_peak_loss = model(prompt, return_type='loss', loss_per_token=True).flatten().cpu()
    with model.hooks([(f'blocks.{LAYER}.mlp.hook_post', snap_to_peak_1)]):
        snap_to_peak_1_loss = model(prompt, return_type='loss', loss_per_token=True).flatten().cpu()
    with model.hooks([(f'blocks.{LAYER}.mlp.hook_post', snap_to_peak_2)]):
        snap_to_peak_2_loss = model(prompt, return_type='loss', loss_per_token=True).flatten().cpu()
    original_loss = model(prompt, return_type='loss', loss_per_token=True).flatten().cpu()

    return [snap_to_closest_peak_loss, snap_to_peak_1_loss, snap_to_peak_2_loss, original_loss]
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
# Filter data for specific prompts
def find_dataset_examples(target: str, data: list[str], model: HookedTransformer, crop: None | tuple[int, int]=None) -> list[str]:
    target_str_tokens = model.to_str_tokens(model.to_tokens(target, prepend_bos=False).flatten())
    print(target_str_tokens)
    target_prompts = []
    for prompt in tqdm(data):
        tokens = model.to_tokens(prompt).flatten()
        str_tokens = model.to_str_tokens(tokens)
        for i in range(len(str_tokens) - len(target_str_tokens) + 1):
            if str_tokens[i:i+len(target_str_tokens)] == target_str_tokens:
                if crop is None:
                    target_prompts.append(prompt)
                else:
                    start_index = max(i-crop[0], 1)
                    end_index = min(i+len(target_str_tokens)+crop[1], len(str_tokens))
                    target_tokens = str_tokens[start_index:end_index]
                    target_prompts.append("".join(target_tokens))

    return target_prompts


target = " in"
examples = find_dataset_examples(target, german_data, model, crop=(15, 1))
#%% 
def snap_pos_to_peak_1(value, hook):
    '''Doesn't snap disabled and ambiguous activations'''
    value[:, -2, NEURON] = 2.5
    return value

def snap_pos_to_peak_2(value, hook):
    '''Doesn't snap disabled and ambiguous activations'''
    value[:, -2, NEURON] = 6.5
    return value

snap_pos_to_peak_1_hook = [(f'blocks.{LAYER}.mlp.hook_post', snap_pos_to_peak_1)]
snap_pos_to_peak_2_hook = [(f'blocks.{LAYER}.mlp.hook_post', snap_pos_to_peak_2)]

mid_word_examples = [example for example in examples if target in example and target+" " not in example]
new_word_examples = [example for example in examples if target+" " in example]
print(len(mid_word_examples), len(new_word_examples))
#%%
# Snap to peak 1 should make the model think that the sentence is continued
for prompt in mid_word_examples[:10]:
    print_prompt(prompt, [(f'blocks.{LAYER}.mlp.hook_post', snap_pos_to_peak_1)], max_value=0.5)
for prompt in new_word_examples[:10]:
    print_prompt(prompt, [(f'blocks.{LAYER}.mlp.hook_post', snap_pos_to_peak_1)], max_value=0.5)

#%%
# Snap to peak 2 should make the model think that it is Klima
for prompt in mid_word_examples[:10]:
    print_prompt(prompt, [(f'blocks.{LAYER}.mlp.hook_post', snap_pos_to_peak_2)], max_value=0.5)
for prompt in new_word_examples[:10]:
    print_prompt(prompt, [(f'blocks.{LAYER}.mlp.hook_post', snap_pos_to_peak_2)], max_value=0.5)

#%% 
def compute_loss_increase(prompts: list[str], model, snapping_hook, pos=-1):
    loss_diffs = []
    for prompt in tqdm(prompts):
        with model.hooks(snapping_hook):
            ablated_loss = model(prompt, return_type='loss', loss_per_token=True).flatten()[pos].item()
        original_loss = model(prompt, return_type='loss', loss_per_token=True).flatten()[pos].item()
        loss_diffs.append(ablated_loss-original_loss)
    return loss_diffs

def plot_snapping_losses(mid_word_examples, new_word_examples, model, snap_pos_to_peak_1_hook, snap_pos_to_peak_2_hook, compute_loss_increase, target):
    mid_word_peak_1_increase = compute_loss_increase(mid_word_examples, model, snap_pos_to_peak_1_hook)
    mid_word_peak_2_increase = compute_loss_increase(mid_word_examples, model, snap_pos_to_peak_2_hook)
    new_word_peak_1_increase = compute_loss_increase(new_word_examples, model, snap_pos_to_peak_1_hook)
    new_word_peak_2_increase = compute_loss_increase(new_word_examples, model, snap_pos_to_peak_2_hook)

    mid_word_peak_1_mean = np.mean(mid_word_peak_1_increase)
    mid_word_peak_2_mean = np.mean(mid_word_peak_2_increase)
    new_word_peak_1_mean = np.mean(new_word_peak_1_increase)
    new_word_peak_2_mean = np.mean(new_word_peak_2_increase)

    mid_word_peak_1_std = np.std(mid_word_peak_1_increase)
    mid_word_peak_2_std = np.std(mid_word_peak_2_increase)
    new_word_peak_1_std = np.std(new_word_peak_1_increase)
    new_word_peak_2_std = np.std(new_word_peak_2_increase)

    z_value = 1.96

    mid_word_peak_1_ci = (mid_word_peak_1_std / np.sqrt(len(mid_word_peak_1_increase))) * z_value
    mid_word_peak_2_ci = (mid_word_peak_2_std / np.sqrt(len(mid_word_peak_2_increase))) * z_value
    new_word_peak_1_ci = (new_word_peak_1_std / np.sqrt(len(new_word_peak_1_increase))) * z_value
    new_word_peak_2_ci = (new_word_peak_2_std / np.sqrt(len(new_word_peak_2_increase))) * z_value

    data = {
        'Peaks': ['Mid_Word_Peak_1', 'New_Word_Peak_1', 'Mid_Word_Peak_2', 'New_Word_Peak_2'],
        'Mean': [mid_word_peak_1_mean, new_word_peak_1_mean, mid_word_peak_2_mean, new_word_peak_2_mean],
        '95% CI': [mid_word_peak_1_ci, new_word_peak_1_ci, mid_word_peak_2_ci, new_word_peak_2_ci]
    }

    df = pd.DataFrame(data)

    fig = px.bar(df, x='Peaks', y='Mean', error_y='95% CI', title=f"Loss Increase for '{target}'")
    fig.show()

# %%
def compute_batched_loss_increase(prompts: list[str], model, snapping_hook, pos=-1):
    with model.hooks(snapping_hook):
        ablated_loss = model(prompts, return_type='loss', loss_per_token=True)[:, pos]
    original_loss = model(prompts, return_type='loss', loss_per_token=True)[:, pos]
    return (ablated_loss-original_loss).tolist()

mid_word_prompt = " seinen Antworten"
new_word_prompt = " seine Antwort auf"

# mid_word_prompt = " im Ausland"
# new_word_prompt = ". Aus dem"

mid_word_prompt = " in der Eurozone"
new_word_prompt = " Millionen Euro sind"
haystack_utils.print_tokenized_word(mid_word_prompt, model), haystack_utils.print_tokenized_word(new_word_prompt, model)
mid_word_prompts = haystack_utils.generate_random_prompts(mid_word_prompt, model, common_tokens, 100, length=20)
new_word_prompts = haystack_utils.generate_random_prompts(new_word_prompt, model, common_tokens, 100, length=20)

plot_snapping_losses(mid_word_prompts, new_word_prompts, model, 
                     snap_pos_to_peak_1_hook, snap_pos_to_peak_2_hook, 
                     compute_batched_loss_increase, target=f'"{mid_word_prompt}" vs "{new_word_prompt}"')
# %%
for prompt in german_data[:5]:
    print_prompt(prompt, [hook_utils.get_ablate_neuron_hook(LAYER, NEURON, 5.5)])
# %%
for prompt in german_data[100:120]:
    print_prompt(prompt, [(f'blocks.{LAYER}.mlp.hook_post', snap_to_peak_2)], max_value=2)
# %%
for prompt in german_data[20:40]:
    print_prompt(prompt, [(f'blocks.{LAYER}.mlp.hook_post', snap_to_peak_1)], max_value=2)
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
def mlp_effects_german(prompt: str, index: int, activate_peak_hooks: list[tuple[str, callable]], 
                       deactivate_peak_hooks: list[tuple[str, callable]], downstream_layers=[9, 10, 11]):
        """Customised to L8 context neuron peaks"""
        
        downstream_components = [(f"blocks.{layer}.hook_{component}_out") for layer in downstream_layers for component in ['mlp', 'attn']]
     
        original, ablated, direct_effect, _ = haystack_utils.get_direct_effect(
                prompt, model, pos=index, context_ablation_hooks=deactivate_peak_hooks, context_activation_hooks=activate_peak_hooks,
                deactivated_components=tuple(downstream_components), activated_components=("blocks.8.hook_mlp_out",))
        
        data = [original, ablated, direct_effect]
        for layer in downstream_layers:
                _, _, _, activated_component_loss = haystack_utils.get_direct_effect(
                        prompt, model, pos=index, context_ablation_hooks=deactivate_peak_hooks, context_activation_hooks=activate_peak_hooks,
                        deactivated_components=tuple(component for component in downstream_components if component != f"blocks.{layer}.hook_mlp_out"),
                        activated_components=(f"blocks.{layer}.hook_mlp_out",))
                data.append(activated_component_loss)
        return data

from typing import Literal
# %%
def pickle_high_loss_prompt_tokens(model, german_data):
    german_losses = []
    for prompt in tqdm(german_data[:200]):
        _, peak_1_loss, peak_2_loss, _ = get_tokenwise_high_loss_diffs(prompt, model)
        diff = (peak_1_loss - peak_2_loss).abs().max()
        german_losses.append(diff)

    index = [i for i in range(len(german_losses))]
    sorted_measure = list(zip(index, german_losses))
    sorted_measure.sort(key=lambda x: x[1], reverse=True)

    high_loss_prompt_tokens: list[tuple[int, int, float, Literal['first', 'second']]] = []
    for i, measure in sorted_measure:
        prompt = german_data[i]
        _, peak_1_loss, peak_2_loss, original_loss = get_tokenwise_high_loss_diffs(prompt, model)
        highest_loss_index = torch.argmax((peak_1_loss - peak_2_loss).abs()).item()
        peak_snapped = 'second' if peak_2_loss[highest_loss_index] > peak_1_loss[highest_loss_index] else 'first'
        high_loss_prompt_tokens.append((i, highest_loss_index, measure, peak_snapped))

    with open(f"data/pythia_160m/high_loss_indices.pkl", "wb") as f:
        pickle.dump(high_loss_prompt_tokens, f)

# pickle_high_loss_prompt_tokens(model, german_data)
# %%
def print_high_loss_prompts(model, german_data):
    with open(f"data/pythia_160m/high_loss_indices.pkl", "rb") as f:
        high_loss_prompt_tokens = pickle.load(f)
    sorted_measure = [(i, measure) for i, _, measure, _ in high_loss_prompt_tokens]
    for i, measure in sorted_measure[:10]:
        prompt = german_data[i]
        tokens = model.to_tokens(prompt)[0]
        str_token_prompt = model.to_str_tokens(tokens)
        closest_loss, peak_1_loss, peak_2_loss, original_loss = get_tokenwise_high_loss_diffs(prompt, model)
        diff = (peak_1_loss - peak_2_loss).abs().flatten().cpu().tolist()
        haystack_utils.clean_print_strings_as_html(str_token_prompt[1:], diff, max_value=3.5, 
                                                additional_measures=[closest_loss, peak_1_loss, peak_2_loss, original_loss], 
                                                additional_measure_names=["closest_loss", "peak_1_loss", "peak_2_loss", "original_loss"])

        highest_loss_index = torch.argmax((peak_1_loss - peak_2_loss).abs()).item()
        if peak_2_loss[highest_loss_index] > peak_1_loss[highest_loss_index]:
            deactivate_final_token_data = mlp_effects_german(prompt, highest_loss_index, [(f'blocks.{LAYER}.mlp.hook_post', snap_to_peak_1)], deactivate_peak_hooks=[(f'blocks.{LAYER}.mlp.hook_post', snap_to_peak_2)])
            haystack_utils.plot_barplot([[item] for item in deactivate_final_token_data],
                                            names=['original', 'ablated', 'direct effect'] + [f'{i}{j}' for j in [9, 10, 11] for i in ["MLP"]], # + ["MLP9 + MLP11"]
                                            title=f'Loss increases from ablating various MLP components at random position, final token')
        else:
            deactivate_other_token_data = mlp_effects_german(prompt, highest_loss_index, [(f'blocks.{LAYER}.mlp.hook_post', snap_to_peak_2)], deactivate_peak_hooks=[(f'blocks.{LAYER}.mlp.hook_post', snap_to_peak_1)])
            haystack_utils.plot_barplot([[item] for item in deactivate_other_token_data], #.cpu().flatten().mean().item()
                                            names=['original', 'ablated', 'direct effect'] + [f'{i}{j}' for j in [9, 10, 11] for i in ["MLP"]], # + ["MLP9 + MLP11"]
                                            title=f'Loss increases from ablating various MLP components at random position, other token')
# %%
print_high_loss_prompts(model, german_data)
# %%
with open(f"data/pythia_160m/high_loss_indices.pkl", "rb") as f:
        high_loss_prompt_tokens = pickle.load(f)
# %%
def final_token_loss_diff(model: HookedTransformer, tokens: torch.Tensor, fwd_hooks: list[tuple[str, callable]]):
    original_loss = model(tokens, return_type='loss', loss_per_token=True)
    with model.hooks(fwd_hooks):
        ablated_loss = model(tokens, return_type='loss', loss_per_token=True)
    # print(ablated_loss.shape, (ablated_loss[: -1] - original_loss[:, -1]).mean(dim=0))
    return (ablated_loss[:, -1] - original_loss[:, -1]).mean(dim=0).cpu()

def plot_truncated_prompt_losses(model: HookedTransformer, fwd_hooks: list[tuple[str, callable]], 
                                 tokens: torch.LongTensor, stop: int | None=None, title: str | None=None) -> None:
    loss_diffs = []
    xticks = []
    stop = stop if stop is not None else tokens.shape[-1]
    for i in range(-2, -stop, -1):
        loss_diffs.append(final_token_loss_diff(model, tokens[..., i:], fwd_hooks))
        xticks.append(str(i))
    plotting_utils.line(loss_diffs, xticks=xticks, xlabel='Starting left index', ylabel='Final token loss', 
                        title=title if title is not None else "Loss diff with different number of tokens before the high loss token")

# %%
# MLP 10 important, snap to peak 2 loss high
i, highest_loss_index, measure, peak_snapped = high_loss_prompt_tokens[0]
tokens = model.to_tokens(german_data[i])
truncated_tokens = tokens[0, :highest_loss_index + 2]
hooks = [(f'blocks.{LAYER}.mlp.hook_post', snap_to_peak_1 if peak_snapped == 'first' else snap_to_peak_2)]
plot_truncated_prompt_losses(model, hooks, truncated_tokens)
plot_truncated_prompt_losses(model, hooks, truncated_tokens, stop=20)

# %%
left_cropped_tokens = truncated_tokens[-16:]
print("".join(model.to_str_tokens(left_cropped_tokens)))
print(final_token_loss_diff(model, left_cropped_tokens, hooks))

str_token_prompt = model.to_str_tokens(tokens)
closest_loss, peak_1_loss, peak_2_loss, original_loss = get_tokenwise_high_loss_diffs(german_data[i], model)
diff = (peak_1_loss - peak_2_loss).abs().flatten().cpu().tolist()
haystack_utils.clean_print_strings_as_html(str_token_prompt[1:], diff, max_value=3.5, 
                                        additional_measures=[closest_loss, peak_1_loss, peak_2_loss, original_loss], 
                                        additional_measure_names=["closest_loss", "peak_1_loss", "peak_2_loss", "original_loss"])

# Activate and deactivate for fewer tokens
# %% 
# I checked whether the loss increases over 16+ tokens for the first high loss increase prompt was 
# due to the L5 context neuron not being fully on and it was unrelated, the model just knows more 
# than trigrams
# Turn context neuron on for both original and ablated
# act_value = haystack_utils.get_mlp_activations(german_data[:200], 5, model, neurons=torch.LongTensor([2649]), mean=True).flatten()
# def activate_l5_context_neuron(value, hook):
#     value[:, :, 2649] = act_value
#     return value
# activate_l5_context_hooks = [('blocks.5.mlp.hook_post', activate_l5_context_neuron)]
# %% 
# Random unigrams are impractical with so many important tokens. I can either swap out tokens
# for random unigrams to find the true minimal reproducible example or I can find real examples
# from the dataset and test them
all_ignore, not_ignore = haystack_utils.get_weird_tokens(model, plot_norms=False)
common_tokens = haystack_utils.get_common_tokens(german_data, model, all_ignore, k=100)

# %%
# Doesn't include all 16 tokens
# end_string = "Europäischen "
for end_string in ["Europäischen Allian", "äischen Allian"]:
    prompt_tokens = haystack_utils.generate_random_prompts(end_string, model, common_tokens)
    plot_truncated_prompt_losses(model, hooks, prompt_tokens, title=end_string)

# %%
print(model.to_str_tokens(left_cropped_tokens[-5:]))
# print("".join(model.to_str_tokens(left_cropped_tokens[-5:])))

euro_prompts = []
for prompt in german_data:
    # if "Europäischen Allian" in prompt:
    if " Allian" in prompt:
        euro_prompts.append(prompt)
print(len(euro_prompts))
# No other data containing the "Europäischen Allian" string

# # %%
# Think about circuits that use both peaks
# check if can make tuple
# %%
import einops
# %%
# Show that peak 2 readers must use a bias scaled from -8.57, 
# and peak 1 readers from -5.7
def plot_direction_norm(model, prompts, layer, neuron):
    '''Not generalizable - need to pass in ranges'''
    direction = model.W_out[layer, neuron]
    final_token_norms = []
    other_token_norms = []
    all_norms = []

    for prompt in prompts:
        _, cache = model.run_with_cache(prompt)
        
        act_values = cache[f'blocks.{layer}.mlp.hook_post'][0, :, neuron]
        direction_out = einops.einsum(direction, act_values, 'd_model, n_acts -> n_acts d_model')

        accumulated_residual = cache.accumulated_resid(layer=9, incl_mid=False, pos_slice=None, return_labels=False)
        pos_means = accumulated_residual[-1, 0, :, :].mean(dim=-1).unsqueeze(-1) # pos d_model -> pos 1

        pos_scaling_factor = cache[f"blocks.{layer + 1}.ln1.hook_scale"].squeeze(0)
        
        # pos_scaling_factor makes it bimodal rather than trimodal
        scaled_direction = ((direction_out - pos_means)/ pos_scaling_factor)
        final_token_scaled_direction = scaled_direction[(act_values > 4.8) & (act_values < 10)]
        other_token_scaled_direction = scaled_direction[(act_values > 0.8) & (act_values < 4.1)]
        final_token_norms.extend(torch.norm(final_token_scaled_direction, dim=-1).cpu().tolist())
        other_token_norms.extend(torch.norm(other_token_scaled_direction, dim=-1).cpu().tolist())
        all_norms.extend(torch.norm(scaled_direction, dim=-1).cpu().tolist())

    fig = px.histogram(other_token_norms, title=f"L{layer}N{neuron} W_out norms when the context neuron is at the first peak")
    fig.show()
    fig = px.histogram(final_token_norms, title="Norms when the context neuron is at the second peak")
    fig.show()
    fig = px.histogram(other_token_norms + final_token_norms, title="Norms when the context neuron is at either peak (excludes ambiguous values)")
    fig.show()
    fig = px.histogram(all_norms, title="All norms")
    fig.show()

plot_direction_norm(model, german_data[:400], layer=8, neuron=2994)

unscaled_peak_1_norm = 5.7
unscaled_peak_2_norm = 8.57
# %%

# For each neuron in L9 and L10, look at its cosine sim with the ctx neuron direction
# and its bias. 
# We know the magnitude (/+- acceptable error) and direction of some information
# The project of the direction onto a reader neuron is the cosine sim * the magnitude, so the necessary bias term
# is the bias term * the cosine sim
# e.g. a reader neuron with a cosine sim of 0.5 gets the direction at half the magnitude

# %%

# required bias term to cut off at peak:
#   unscaled_peak_2_norm * cosine_sim(neuron_in, ctx_neuron_direction)
#   
# to get anything above the mid point between peaks:
def get_context_reader_properties_df(model: HookedTransformer, unscaled_peak_1_norm, unscaled_peak_2_norm, layer=9) -> pd.DataFrame:
    """Returns a dataframe with columns: context neuron on, reader neuron on, reader neuron index
    Only reliable for layer 9 because it needs extra layer norm processing otherwise"""
    cosine_sim = torch.nn.CosineSimilarity(dim=0)
    l8_n2994_direction = model.W_out[8, 2994]
    result = []
    for neuron in range(model.cfg.d_mlp):
        sim = cosine_sim(l8_n2994_direction, model.W_in[layer, :, neuron]).item()

        unscaled_peak_midpoint = (unscaled_peak_1_norm + unscaled_peak_2_norm) / 2
        scaled_peak_midpoint = unscaled_peak_midpoint * sim        

        neuron_bias = model.b_in[layer, neuron].item()

        bias_diff_from_midpoint = abs(scaled_peak_midpoint - neuron_bias)

        # to select peak 2 use a bias of cosine_sim * -8.57
        result.append([
            sim, 
            neuron_bias, 
            scaled_peak_midpoint,
            bias_diff_from_midpoint,
            unscaled_peak_1_norm * sim,
            unscaled_peak_2_norm * sim])

    return pd.DataFrame(result, columns=["cosine_sim", "bias", "peak_midpoint_norm", "bias_diff_from_peak_midpoint", "scaled_peak_1_norm", "scaled_peak_2_norm"])
# %%
df = get_context_reader_properties_df(model, unscaled_peak_1_norm, unscaled_peak_2_norm, layer=9)
print(df.sort_values(['cosine_sim'], ascending=True).head())
print(df.sort_values(['cosine_sim'], ascending=False).head())
# %%
px.histogram(df['cosine_sim']).show()
print(df[(df['cosine_sim'] > 0.05)].sort_values(['bias_diff_from_peak_midpoint'], ascending=True))
# Many neurons take the peak 2 range with cosine sims < 0.06
# Time to do neuron on/off stats

# %% 
def get_context_reader_activations_df(model: HookedTransformer, data: list[str]) -> pd.DataFrame:
    result = [] # context neuron on or off, downstream neuron on or off, neuron index
    
    for prompt in tqdm(data):
        _, cache = model.run_with_cache(prompt)
        context_neuron_acts = cache[f'blocks.{LAYER}.mlp.hook_post'][0, 5:, NEURON]
        for neuron in range(model.cfg.d_mlp):
            neuron_act = cache[f'blocks.{LAYER+1}.mlp.hook_post'][0, 5:, neuron]
            context_neuron_peak_1 = (context_neuron_acts > 0.8) & (context_neuron_acts < 4.1)
            context_neuron_peak_2 = (context_neuron_acts > 4.8) & (context_neuron_acts < 10)
            result.extend(zip([neuron]*len(neuron_act), context_neuron_peak_1.tolist(), context_neuron_peak_2.tolist(), neuron_act.tolist()))

    return pd.DataFrame(result, columns=["neuron", "context_neuron_peak_1", "context_neuron_peak_2", "neuron_act"])

act_df = get_context_reader_activations_df(model, german_data[:100])
# %%
grouped = act_df.groupby('neuron')
df['peak_1_fire_rate'] = grouped.apply(
    lambda x: 
        ((x['neuron_act'] > 0) & x['context_neuron_peak_1']).sum() / 
        ((x['neuron_act'] > 0).sum() + 1e-10))

df['peak_2_fire_rate'] = grouped.apply(
    lambda x: 
        ((x['neuron_act'] > 0) & x['context_neuron_peak_2']).sum() / 
        ((x['neuron_act'] > 0).sum() + 1e-10))
# Figure out whether there's a general boost of many German tokens middle of word / new word 
# and/or an isolated circuit per prompt (e.g. an AND gate)
# %%
cols_of_interest = ["cosine_sim", "bias", "peak_1_fire_rate", "peak_2_fire_rate"]
rows = df['bias_diff_from_peak_midpoint'].nsmallest(5).index
print(df.loc[rows, cols_of_interest]) # cosine sim of 0.05 and bias cuts off mid peak 
rows = df['cosine_sim'].nlargest(5).index
print(df.loc[rows, cols_of_interest]) # high pos cosine sims
rows = df['cosine_sim'].nsmallest(5).index
print(df.loc[rows, cols_of_interest]) # high neg cosine sims
# %%
print(df.loc[df['peak_2_fire_rate'].nlargest(5).index, cols_of_interest])

# %%
rows = df['cosine_sim'].nsmallest(100).index
print(df.loc[rows, cols_of_interest]) # high neg cosine sims

# %%
with open(f"data/pythia_160m/reader_neurons.pkl", "wb") as f:
    pickle.dump(df, f)
with open(f"data/pythia_160m/reader_neurons.pkl", "rb") as f:
    reader_neurons = pickle.load(f)
# %%

print("Outliers")
print(df.loc[[908, 890, 1473], cols_of_interest]) # high neg cosine sims
# %%

# %%
def get_context_reader_activations_df(model: HookedTransformer, data: list[str], layer=9) -> pd.DataFrame:
    result = [] # context neuron on or off, downstream neuron on or off, neuron index
    
    for prompt in tqdm(data):
        _, cache = model.run_with_cache(prompt)
        context_neuron_acts = cache[f'blocks.{LAYER}.mlp.hook_post'][0, 5:, NEURON]
        for neuron in range(model.cfg.d_mlp):
            neuron_act = cache[f'blocks.{layer}.mlp.hook_post'][0, 5:, neuron]
            context_neuron_peak_1 = (context_neuron_acts > 0.8) & (context_neuron_acts < 4.1)
            context_neuron_peak_2 = (context_neuron_acts > 4.8) & (context_neuron_acts < 10)
            result.extend(zip([neuron]*len(neuron_act), context_neuron_peak_1.tolist(), context_neuron_peak_2.tolist(), neuron_act.tolist()))

    return pd.DataFrame(result, columns=["neuron", "context_neuron_peak_1", "context_neuron_peak_2", "neuron_act"])

df_11 = get_context_reader_activations_df(model, german_data[:100], layer=11)
# %%
grouped = df_11.groupby('neuron')

peak_1_precision = grouped.apply(
    lambda x: 
        ((x['neuron_act'] > 0) & x['context_neuron_peak_1']).sum() / 
        (((x['neuron_act'] > 0)).sum() + 1e-10))
peak_2_precision = grouped.apply(
    lambda x: 
        ((x['neuron_act'] > 0) & x['context_neuron_peak_2']).sum() / 
        ((x['neuron_act'] > 0).sum() + 1e-10))
peak_1_precision_between_peaks = grouped.apply(
    lambda x: 
        ((x['neuron_act'] > 0) & x['context_neuron_peak_1']).sum() / 
        (((x['neuron_act'] > 0) & (x['context_neuron_peak_1'] | x['context_neuron_peak_2'])).sum() + 1e-10))
peak_2_precision_between_peaks = grouped.apply(
    lambda x: 
        ((x['neuron_act'] > 0) & x['context_neuron_peak_2']).sum() / 
        ((x['neuron_act'] > 0 & (x['context_neuron_peak_1'] | x['context_neuron_peak_2'])).sum() + 1e-10))
peak_1_fire_rate = grouped.apply(
    lambda x: 
        ((x['neuron_act'] > 0) & x['context_neuron_peak_1']).sum() / 
        (((x['context_neuron_peak_1'])).sum() + 1e-10))
peak_2_fire_rate = grouped.apply(
    lambda x: 
        ((x['neuron_act'] > 0) & x['context_neuron_peak_2']).sum() / 
        ((x['context_neuron_peak_2']).sum() + 1e-10))
firing_rate = grouped.apply(
    lambda x:
        ((x['neuron_act'] > 0)).sum() / 
        ((x['neuron_act'] > float("-inf")).sum() + 1e-10))

firing_rate_df = pd.DataFrame({
    "peak_1_precision": peak_1_precision,
    "peak_2_precision": peak_2_precision,
    "peak_1_precision_between_peaks": peak_1_precision_between_peaks,
    "peak_2_precision_between_peaks": peak_2_precision_between_peaks,
    "peak_1_fire_rate": peak_1_fire_rate,
    "peak_2_fire_rate": peak_2_fire_rate,
    "firing_rate": firing_rate,
    })

print(firing_rate_df.head())

# %%

firing_rate_df["x"] = grouped.apply(
    lambda x: 
        ((x['neuron_act'] > 0) & x['context_neuron_peak_1']).sum() / 
        (((x['neuron_act'] > 0)).sum() + 1e-10))
firing_rate_df["y"] = grouped.apply(
    lambda x: 
        ((x['neuron_act'] > 0) & x['context_neuron_peak_2']).sum() / 
        (((x['neuron_act'] > 0).sum() + 1e-10)))

# %%

plotting_utils.plot_neuron_acts(model, german_data, [[11, 678]])
# %%
px.scatter(firing_rate_df, x="peak_1_fire_rate", y="peak_2_fire_rate", hover_data=firing_rate_df.columns)

# %%
print(firing_rate_df.loc[firing_rate_df['peak_1_fire_rate'].nlargest(5).index])
print(firing_rate_df.loc[firing_rate_df['peak_2_fire_rate'].nlargest(5).index])
# %%
px.scatter(firing_rate_df, x="x", y="y", hover_data=firing_rate_df.columns)
# %%
px.scatter(firing_rate_df[firing_rate_df["firing_rate"] > 0.1], x="peak_1_precision", y="peak_2_precision", hover_data=firing_rate_df.columns)
# %%
px.scatter(firing_rate_df[firing_rate_df["firing_rate"] > 0.1], x="peak_1_precision_between_peaks", y="peak_2_precision_between_peaks", hover_data=firing_rate_df.columns)

# %%
cols_of_interest = ["x", "y", "peak_1_precision", "peak_2_precision", "peak_1_fire_rate", "peak_2_fire_rate"]
print("Largest peak 1 precisions:")
print(firing_rate_df[firing_rate_df["firing_rate"] > 0.1].loc[firing_rate_df[firing_rate_df["firing_rate"] > 0.1]['peak_1_precision'].nlargest(5).index, cols_of_interest])
print("Largest peak 2 precisions:")
print(firing_rate_df[firing_rate_df["firing_rate"] > 0.1].loc[firing_rate_df[firing_rate_df["firing_rate"] > 0.1]['peak_2_precision'].nlargest(5).index, cols_of_interest])
print("Largest peak 1 fire rates:")
print(firing_rate_df[firing_rate_df["firing_rate"] > 0.1].loc[firing_rate_df[firing_rate_df["firing_rate"] > 0.1]['peak_1_fire_rate'].nlargest(5).index, cols_of_interest])
print("Largest peak 2 fire rates:")
print(firing_rate_df[firing_rate_df["firing_rate"] > 0.1].loc[firing_rate_df[firing_rate_df["firing_rate"] > 0.1]['peak_2_fire_rate'].nlargest(5).index, cols_of_interest])
# %%


