
#%% 

import pickle
from typing import Literal
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer, ActivationCache, utils
from jaxtyping import Float, Int, Bool
from torch import Tensor
from tqdm.auto import tqdm
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display, clear_output
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
import plotting_utils
import hook_utils
import plotly.express as px


pio.renderers.default = "notebook_connected"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)

import haystack_utils

%reload_ext autoreload
%autoreload 2
#%%

model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device)

activate_neurons_fwd_hooks, deactivate_neurons_fwd_hooks = haystack_utils.get_context_ablation_hooks(3, [669], model)
all_ignore, _ = haystack_utils.get_weird_tokens(model, plot_norms=False)

german_data = haystack_utils.load_json_data("data/german_europarl.json")[:200]
common_tokens = haystack_utils.get_common_tokens(german_data, model, all_ignore, k=100)

# Sort tokens into new word vs continuation
new_word_tokens = []
continuation_tokens = []
for token in common_tokens:
    str_token = model.to_single_str_token(token.item())
    if str_token.startswith(" "):
        new_word_tokens.append(token)
    else:
        continuation_tokens.append(token)
new_word_tokens = torch.stack(new_word_tokens)
continuation_tokens = torch.stack(continuation_tokens)

# %%
# SETUP
option = " deinen Vorschlägen"
prompts = haystack_utils.generate_random_prompts(option, model, common_tokens, 500, length=20).cpu()
normalize_diffs = False
hook_name = "hook_post"
type = "logits"
# %%
data = haystack_utils.compute_and_conditions(prompts, model, option, type=type, common_tokens=common_tokens, activate_context_hook=activate_neurons_fwd_hooks, deactivate_context_hooks=deactivate_neurons_fwd_hooks)
data["Prompt"] = [option]
and_df = pd.DataFrame(data).round(2)
#%%
features = ["Fix Current", "Fix Previous", "Fix Context", "Single Feature", "Two Features", "Merge Tokens"]
df_melted = and_df[features].melt(var_name='Category', value_name=type + ' difference')
fig = px.bar(df_melted, x='Category', y=type + ' difference', title=f'AND criteria {type} differences')
fig.show()
# %% 
# Get average activation for each neuron on random German prompts
pre_act = haystack_utils.get_mlp_activations(german_data, 5, model, 200, hook_pre=True, mean=False)
post_act = haystack_utils.get_mlp_activations(german_data, 5, model, 200, hook_pre=False, mean=False)

# %%
def normalize_df(df, hook_name):
    df = df.copy()
    activations = pre_act if hook_name == "hook_pre" else post_act
    std = activations.std(axis=0).cpu().numpy()
    columns = ["YYY", "YYN", "YNY", "NYY", "YNN", "NYN", "NNY", "NNN"]
    for col in columns:
        df[col] = df[col] / std
    return df

unscaled_df = haystack_utils.activation_data_frame(option, prompts, model, common_tokens, activate_neurons_fwd_hooks, deactivate_neurons_fwd_hooks, mlp_hook=hook_name)
unscaled_df = haystack_utils.compute_and_feature_diffs(unscaled_df)
if normalize_diffs:
    df = normalize_df(unscaled_df, hook_name)
    haystack_utils.compute_and_feature_diffs(df)
else:
    df = unscaled_df.copy()

# %%
feature = "Greater Positive"
num_neurons = 50
# include_modes = ["All Positive", "Greater Positive", "All Negative", "Smaller Negative"]

all_losses = haystack_utils.get_and_neuron_ablation_losses(prompts, model, df, num_neurons, include_mode=feature,
                                            context_ablation_hooks=deactivate_neurons_fwd_hooks, context_activation_hooks=activate_neurons_fwd_hooks)

names = list(all_losses.keys())
short_names = [name.split(" ")[0] for name in names]
loss_values = [[all_losses[name]] for name in names]
plot = plotting_utils.plot_barplot(loss_values, names, short_names=short_names,
                            ylabel="Last token loss",
                            title=f"Loss increase when patching groups of neurons (ablation mode: YYN)",
                            width=750, show=False)
plot.show()
# %%
