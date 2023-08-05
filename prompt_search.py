# %%

import torch
import numpy as np
from torch import einsum
from tqdm.auto import tqdm
import seaborn as sns
from transformer_lens import HookedTransformer, ActivationCache, utils, patching
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
import functools
from transformer_lens.hook_points import HookPoint
# import circuitsvis
from IPython.display import HTML
from plotly.express import line
import plotly.express as px
from tqdm.auto import tqdm
import json
import gc
import plotly.graph_objects as go

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from plotly.subplots import make_subplots
# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio
pio.renderers.default = "notebook_connected"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)

from haystack_utils import load_txt_data, get_mlp_activations, line
import haystack_utils

%reload_ext autoreload
%autoreload 2

# %% 

model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device)

german_data = haystack_utils.load_json_data("data/german_europarl.json")
english_data = haystack_utils.load_json_data("data/english_europarl.json")


english_activations = {}
german_activations = {}
for layer in range(3, 6):
    english_activations[layer] = get_mlp_activations(english_data[:200], layer, model, mean=False)
    german_activations[layer] = get_mlp_activations(german_data[:200], layer, model, mean=False)

all_ignore, _ = haystack_utils.get_weird_tokens(model, plot_norms=False)
common_tokens = haystack_utils.get_common_tokens(german_data, model, all_ignore, k=100)


LOG_PROB_THRESHOLD = -7
LAYER_TO_ABLATE = 3
NEURONS_TO_ABLATE = [669]
MEAN_ACTIVATION_ACTIVE = german_activations[LAYER_TO_ABLATE][:, NEURONS_TO_ABLATE].mean()
MEAN_ACTIVATION_INACTIVE = english_activations[LAYER_TO_ABLATE][:, NEURONS_TO_ABLATE].mean()

def deactivate_neurons_hook(value, hook):
    value[:, :, NEURONS_TO_ABLATE] = MEAN_ACTIVATION_INACTIVE
    return value
deactivate_neurons_fwd_hooks=[(f'blocks.{LAYER_TO_ABLATE}.mlp.hook_post', deactivate_neurons_hook)]

def activate_neurons_hook(value, hook):
    value[:, :, NEURONS_TO_ABLATE] = MEAN_ACTIVATION_ACTIVE
    return value
activate_neurons_fwd_hooks=[(f'blocks.{LAYER_TO_ABLATE}.mlp.hook_post', activate_neurons_hook)]
# %% 
news_data = haystack_utils.load_txt_data("german_news.csv")
german_news_data = []
for example in tqdm(news_data[:-1]):
    index = example.index(";")
    example = example[index+1:]
    if len(example) > 500:
        german_news_data.append(example[:min(len(example), 2000)])

print(len(german_news_data))


# %% 
DATA = german_data
german_losses = []
deactivated_components=("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.5.hook_mlp_out")
activated_components=("blocks.4.hook_mlp_out",)

for prompt in tqdm(DATA[:200]):
    original_loss, ablated_loss, context_and_activated_loss, only_activated_loss = \
    haystack_utils.get_direct_effect(prompt, model, pos=None,
        activated_components=activated_components, deactivated_components=deactivated_components,
        context_ablation_hooks=deactivate_neurons_fwd_hooks, context_activation_hooks=activate_neurons_fwd_hooks)
    german_losses.append((original_loss, ablated_loss, context_and_activated_loss, only_activated_loss))
# %%

# def interest_measure(original_loss, ablated_loss, context_and_activated_loss, only_activated_loss):
#     loss_diff = (ablated_loss - original_loss) # High ablation loss increase
#     mlp_5_power = (only_activated_loss - original_loss) # High loss increase from MLP5
#     mlp_5_power[mlp_5_power < 0] = 0
#     percent_explained = mlp_5_power / loss_diff
#     percent_explained[percent_explained < 0] = 0
#     percent_explained[percent_explained > 1] = 1
#     percent_explained = 1 - percent_explained
#     combined = loss_diff * percent_explained
#     combined[original_loss > 4] = 0
#     combined[original_loss > ablated_loss] = 0
#     return combined


def interest_measure(original_loss, ablated_loss, context_and_activated_loss, only_activated_loss):
    loss_diff = (ablated_loss - original_loss) # High ablation loss increase
    direct_effect = (ablated_loss - only_activated_loss) # High loss increase from MLP5
    min_effect = torch.min(loss_diff, direct_effect)
    min_effect[original_loss > 4] = 0
    return min_effect

# def interest_measure(original_loss, ablated_loss, context_and_activated_loss, only_activated_loss):
#     loss_diff = (ablated_loss - original_loss) # High ablation loss increase
#     direct_effect = (context_and_activated_loss - original_loss) # High loss increase from MLP5
#     min_effect = torch.min(loss_diff, direct_effect)
#     min_effect[original_loss > 4] = 0
#     return min_effect

def get_mlp5_decrease_measure(losses: list[tuple[Float[Tensor, "pos"], Float[Tensor, "pos"], Float[Tensor, "pos"], Float[Tensor, "pos"]]]):
    measure = []
    for original_loss, ablated_loss, context_and_activated_loss, only_activated_loss in losses:
        combined = interest_measure(original_loss, ablated_loss, context_and_activated_loss, only_activated_loss)
        measure.append(combined.max().item())
    return measure

measure = get_mlp5_decrease_measure(german_losses)
index = [i for i in range(len(measure))]

sorted_measure = list(zip(index, measure))
sorted_measure.sort(key=lambda x: x[1], reverse=True)

def print_prompt(prompt: str):
    str_token_prompt = model.to_str_tokens(model.to_tokens(prompt))
    original_loss, ablated_loss, context_and_activated_loss, only_activated_loss = \
        haystack_utils.get_direct_effect(prompt, model, pos=None, 
                                         activated_components=activated_components, deactivated_components=deactivated_components,
                                         context_ablation_hooks=deactivate_neurons_fwd_hooks, context_activation_hooks=activate_neurons_fwd_hooks)

    pos_wise_diff = interest_measure(original_loss, ablated_loss, context_and_activated_loss, only_activated_loss).flatten().cpu().tolist()

    loss_list = [loss.flatten().cpu().tolist() for loss in [original_loss, ablated_loss, context_and_activated_loss, only_activated_loss]]
    loss_names = ["original_loss", "ablated_loss", "context_and_activated_loss", "only_activated_loss"]
    haystack_utils.clean_print_strings_as_html(str_token_prompt[1:], pos_wise_diff, max_value=5, additional_measures=loss_list, additional_measure_names=loss_names)

# %% 
for i, measure in sorted_measure[30:40]:
    print_prompt(DATA[i])

#%%
def search_for_token_examples(token, data, model):
    example_prompts = []
    for prompt in data:
        str_tokens = model.to_str_tokens(model.to_tokens(prompt).flatten())
        if token in str_tokens:
            prompt_index = str_tokens.index(token)
            print(str_tokens[max(0, prompt_index-3):min(len(str_tokens), prompt_index+2)])


search_for_token_examples(" Pr", english_data, model)


# %%

prompt = " Ich bin der Ansicht"
haystack_utils.get_boosted_tokens(prompt, model, deactivate_neurons_fwd_hooks, all_ignore)

#%%
haystack_utils.print_tokenized_word(" Ansicht", model)








# %%

ngram = " statt"
ngram_str_tokens = model.to_str_tokens(model.to_tokens(ngram, prepend_bos=False))
prompts = haystack_utils.generate_random_prompts(ngram, model, common_tokens, 100)


#  %%

average_loss_plot = haystack_utils.get_average_loss_plot_method(
    activate_neurons_fwd_hooks, deactivate_neurons_fwd_hooks, "MLP5")

average_loss_plot(prompts, model, token=ngram_str_tokens, plot=True)

# %%

mlp4_loss_plot = haystack_utils.get_average_loss_plot_method(
    activate_neurons_fwd_hooks, deactivate_neurons_fwd_hooks, "MLP4",
    deactivated_components = ("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.5.hook_mlp_out", ),
    activated_components = ("blocks.4.hook_mlp_out",  ))

mlp4_loss_plot(prompts, model, token=ngram_str_tokens, plot=True)
# %%
mlp4_mlp5_loss_plot = haystack_utils.get_average_loss_plot_method(
    activate_neurons_fwd_hooks, deactivate_neurons_fwd_hooks, "MLP4 + MLP5",
    deactivated_components = ("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", ),
    activated_components = ("blocks.4.hook_mlp_out", "blocks.5.hook_mlp_out"))

mlp4_mlp5_loss_plot(prompts, model, token=ngram_str_tokens, plot=True)
# %%

context_only_loss_plot = haystack_utils.get_average_loss_plot_method(
    activate_neurons_fwd_hooks, deactivate_neurons_fwd_hooks, "None",
    deactivated_components = ("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out", "blocks.5.hook_mlp_out"),
    activated_components = ())

context_only_loss_plot(prompts, model, token=ngram_str_tokens, plot=True)
# %%

answer_token = model.to_single_token(ngram_str_tokens[-1])
print(answer_token)

# %%
average_loss_plot = haystack_utils.get_average_loss_plot_method(
    activate_neurons_fwd_hooks, deactivate_neurons_fwd_hooks, "MLP5", return_type="logits", answer_token=answer_token)

average_loss_plot(prompts, model, token=ngram_str_tokens, plot=True)

# %%

mlp4_loss_plot = haystack_utils.get_average_loss_plot_method(
    activate_neurons_fwd_hooks, deactivate_neurons_fwd_hooks, "MLP4",
    deactivated_components = ("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.5.hook_mlp_out", ),
    activated_components = ("blocks.4.hook_mlp_out",  ), return_type="logits", answer_token=answer_token)

mlp4_loss_plot(prompts, model, token=ngram_str_tokens, plot=True)
# %%
mlp4_mlp5_loss_plot = haystack_utils.get_average_loss_plot_method(
    activate_neurons_fwd_hooks, deactivate_neurons_fwd_hooks, "MLP4 + MLP5",
    deactivated_components = ("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", ),
    activated_components = ("blocks.4.hook_mlp_out", "blocks.5.hook_mlp_out"), return_type="logits", answer_token=answer_token)

mlp4_mlp5_loss_plot(prompts, model, token=ngram_str_tokens, plot=True)
# %%

context_only_loss_plot = haystack_utils.get_average_loss_plot_method(
    activate_neurons_fwd_hooks, deactivate_neurons_fwd_hooks, "None",
    deactivated_components = ("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out", "blocks.5.hook_mlp_out"),
    activated_components = (), return_type="logits", answer_token=answer_token)

context_only_loss_plot(prompts, model, token=ngram_str_tokens, plot=True)

# %%
with model.hooks(fwd_hooks=deactivate_neurons_fwd_hooks):
    utils.test_prompt(" diese Ans", "icht", model, prepend_space_to_answer=False)
# %%
