# %%
import torch
import numpy as np
from torch import einsum
from tqdm.auto import tqdm
import seaborn as sns
from transformer_lens import HookedTransformer, ActivationCache, utils
from datasets import load_dataset
from einops import einsum
import pandas as pd
from transformer_lens import utils
from rich.table import Table, Column
from rich import print as rprint
from jaxtyping import Float, Int, Bool
from typing import List, Tuple
from torch import Tensor
import einops
from transformer_lens.hook_points import HookPoint
from IPython.display import HTML
import plotly.express as px
from tqdm.auto import tqdm
# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio
pio.renderers.default = "notebook_connected"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)

import haystack_utils
import hook_utils

%reload_ext autoreload
%autoreload 2
# %%

haystack_utils.clean_cache()
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device)

german_data = haystack_utils.load_json_data("data/german_europarl.json")[:200]
english_data = haystack_utils.load_json_data("data/english_europarl.json")[:200]

all_ignore, not_ignore = haystack_utils.get_weird_tokens(model, plot_norms=False)
common_tokens = haystack_utils.get_common_tokens(german_data, model, all_ignore, k=100)
# %%
DATA = german_data

# %%

context_neurons = [(3, 669)]
#context_neurons = [(5, 2649), (8,2994)]

mean_active_activations = haystack_utils.get_mean_neuron_activations(context_neurons, german_data, model)
mean_inactive_activations = haystack_utils.get_mean_neuron_activations(context_neurons, english_data, model)

# %%
activate_context_hooks = hook_utils.get_ablate_context_neurons_hooks(context_neurons, mean_active_activations)
deactivate_context_hooks = hook_utils.get_ablate_context_neurons_hooks(context_neurons, mean_inactive_activations)


# %%
highest_context_layer = max([x[0] for x in context_neurons])
highest_layer = model.cfg.n_layers - 1

activated_mlp_layers = [4]#[highest_layer]
activated_components = [f"blocks.{layer}.hook_mlp_out" for layer in activated_mlp_layers]
deactivated_components = [f"blocks.{layer}.hook_mlp_out" for layer in range(highest_context_layer + 1, highest_layer + 1) if layer not in activated_mlp_layers]
deactivated_components += [f"blocks.{layer}.hook_attn_out" for layer in range(highest_context_layer+1, highest_layer+1)]

# %%
german_losses = []
for prompt in tqdm(DATA[:100]):
    original_loss, ablated_loss, context_and_activated_loss, only_activated_loss = haystack_utils.get_direct_effect(prompt, model, pos=None, 
                                                                                                                    context_ablation_hooks=deactivate_context_hooks, context_activation_hooks=activate_context_hooks,
                                                                                                                    activated_components=activated_components, deactivated_components=deactivated_components)
    german_losses.append((original_loss, ablated_loss, context_and_activated_loss, only_activated_loss))

# %%

def interest_measure(original_loss, ablated_loss, context_and_activated_loss, only_activated_loss):
    loss_diff = (ablated_loss - original_loss) # High ablation loss increase
    mlp_power = (only_activated_loss - original_loss) # High loss increase from MLP5
    mlp_power[mlp_power < 0] = 0
    percent_explained = mlp_power / loss_diff
    percent_explained[percent_explained < 0] = 0
    percent_explained[percent_explained > 1] = 1
    percent_explained = 1 - percent_explained
    combined = loss_diff * percent_explained
    combined[original_loss > 4] = 0
    combined[only_activated_loss > 5] = 0
    combined[original_loss > ablated_loss] = 0
    return combined

def sort_losses_by_interest(losses: list[tuple[Float[Tensor, "pos"], Float[Tensor, "pos"], Float[Tensor, "pos"], Float[Tensor, "pos"]]], interest_measure: callable) -> list[tuple[int, float]]:
    measure = []
    pos_wise_measure = []
    for original_loss, ablated_loss, context_and_activated_loss, only_activated_loss in losses:
        combined = interest_measure(original_loss, ablated_loss, context_and_activated_loss, only_activated_loss)
        measure.append(combined.max().item())
        pos_wise_measure.append(combined)

    index = [i for i in range(len(measure))]
    sorted_measure = list(zip(index, measure, losses, pos_wise_measure))
    sorted_measure.sort(key=lambda x: x[1], reverse=True)
    return sorted_measure

def print_prompt(measure, data):
    index, measure, losses, pos_wise_measure = measure
    original_loss, ablated_loss, context_and_activated_loss, only_activated_loss = losses
    prompt = data[index]
    str_token_prompt = model.to_str_tokens(model.to_tokens(prompt))
    loss_list = [loss.flatten().cpu().tolist() for loss in [original_loss, ablated_loss, context_and_activated_loss, only_activated_loss]]
    loss_names = ["original_loss", "ablated_loss", "context_and_activated_loss", "only_activated_loss"]
    haystack_utils.clean_print_strings_as_html(str_token_prompt[1:], pos_wise_measure, max_value=5, additional_measures=loss_list, additional_measure_names=loss_names)

# %%
sorted_measure = sort_losses_by_interest(german_losses, interest_measure)

# %%
for i in range(1):
    print_prompt(sorted_measure[i], DATA)
# %%

print(sorted_measure[0])
# %%
index = sorted_measure[0][0]

with model.hooks(fwd_hooks=activate_context_hooks):
    loss = model(DATA[index], return_type="loss", loss_per_token=True)

print(loss)

import plotly.express as px
px.histogram(loss.flatten().cpu().numpy())
# %%
