
#%% 

from typing import Literal
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer, ActivationCache, utils, patching
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

from haystack_utils import get_mlp_activations
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

context_direction = model.W_out[3, 669, :]
# %%
def compute_and_conditions(option, type: Literal["logits", "loss"]):
    ANSWER_TOKEN_ID = model.to_tokens(option).flatten()[-1].item()
    prompts = haystack_utils.generate_random_prompts(option, model, common_tokens, 1000, length=20)
    multiplier = 1 if type == "loss" else -1

    def get_value(prompts):
        if type == "logits":
            value = model(prompts, return_type="logits", loss_per_token=True)[:, -2, ANSWER_TOKEN_ID].mean().item()
        elif type == "loss":
            value = model(prompts, return_type="loss", loss_per_token=True)[:, -1].mean().item()
        return value
    
    # COMPUTE AND CONDITIONS
    # Fix current token
    ablated_prompts_ny = haystack_utils.create_ablation_prompts(prompts, "NYY", common_tokens)
    ablated_prompts_yn = haystack_utils.create_ablation_prompts(prompts, "YNY", common_tokens)
    ablated_prompts_nn = haystack_utils.create_ablation_prompts(prompts, "NNY", common_tokens)

    yyy_value = get_value(prompts)
    nyy_value = get_value(ablated_prompts_ny)
    yny_value = get_value(ablated_prompts_yn)
    nny_value = get_value(ablated_prompts_nn)

    with model.hooks(fwd_hooks=deactivate_neurons_fwd_hooks):
        yyn_value = get_value(prompts)
        nyn_value = get_value(ablated_prompts_ny)
        ynn_value = get_value(ablated_prompts_yn)
        nnn_value = get_value(ablated_prompts_nn)

    # Fix current token
    yyn_nyn_diff = (nyn_value - yyn_value) * multiplier
    nyy_nyn_diff = (nyn_value - nyy_value) * multiplier
    yyy_nyn_diff =  (nyn_value - yyy_value) * multiplier
    current_diffs = yyy_nyn_diff - (yyn_nyn_diff + nyy_nyn_diff)

    # Group current and previous token
    yyn_nnn_diff = (nnn_value - yyn_value) * multiplier
    nny_nnn_diff = (nnn_value - nny_value) * multiplier 
    yyy_nnn_diff =  (nnn_value - yyy_value) * multiplier
    grouped_diffs = yyy_nnn_diff - (yyn_nnn_diff + nny_nnn_diff)

    # All groups of two features
    nyy_nnn_diff = (nnn_value - nyy_value) * multiplier
    yny_nnn_diff = (nnn_value - yny_value) * multiplier
    yyn_nnn_diff = (nnn_value - yyn_value) * multiplier
    # Individual features
    nny_nnn_diff = (nnn_value - nny_value) * multiplier
    nyn_nnn_diff = (nnn_value - nyn_value) * multiplier
    ynn_nnn_diff = (nnn_value - ynn_value) * multiplier
    # Loss increase for all features
    yyy_nnn_diff = (nnn_value - yyy_value) * multiplier

    individual_diffs = yyy_nnn_diff - (nny_nnn_diff + nyn_nnn_diff + ynn_nnn_diff)
    two_feature_diffs = yyy_nnn_diff - (nyy_nnn_diff + yny_nnn_diff + yyn_nnn_diff)/2

    result = {
        "yyy": yyy_value,
        "nny": nny_value,
        "nyn": nyn_value,
        "ynn": ynn_value,
        "nyy": nyy_value,
        "yny": yny_value,
        "yyn": yyn_value,
        "nnn": nnn_value,
        "current_token_diffs": current_diffs,
        "grouped_token_diffs": grouped_diffs,
        "individiual_features_diffs": individual_diffs,
        "two_features_diffs": two_feature_diffs,
    }

    return result

# %%
options = ["orschlägen", " häufig", " beweglich"]
all_res = {}
for option in options:
    all_res[option] = {}
    for type in ["loss", "logits"]:
        result = compute_and_conditions(option, type)
        all_res[option][type] = result
        
# %%
with open("data/and_neurons/and_conditions.json", "w") as f:
    json.dump(all_res, f, indent=4)
# %%
