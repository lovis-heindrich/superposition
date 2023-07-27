
#%% 

import pickle
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

haystack_utils.clean_cache()
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
#%%
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


# %%
# all_prompts = {}
# for option in tqdm(options):
#     # Create global prompts
#     option_prompts = haystack_utils.generate_random_prompts(option, model, common_tokens, 5000, length=20).cpu()
#     all_prompts[option] = option_prompts

# Only overwrite if really needed!
# with open("data/prompts.pkl", "wb") as f:
#     pickle.dump(all_prompts, f)

with open("data/prompts.pkl", "rb") as f:
    all_prompts = pickle.load(f)

# %% 

# SCALING

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


dfs = {}
for option in tqdm(options):
    dfs[option] = {}
    prompts = all_prompts[option][:2000]
    for hook_name in ["hook_pre", "hook_post"]:
        dfs[option][hook_name] = {}
        for scale in [True, False]:
            df = haystack_utils.activation_data_frame(option, prompts, model, common_tokens, deactivate_neurons_fwd_hooks, mlp_hook=hook_name)
            if scale:
                df = normalize_df(df, hook_name)
            dfs[option][hook_name]["Scaled" if scale else "Unscaled"] = df
# %%
with open("data/and_neurons/activation_dfs.pkl", "wb") as f:
    pickle.dump(dfs, f)
# %%


for option in options:
    for hook_name in ["hook_pre", "hook_post"]:
        for scale in [True, False]:
            df = dfs[option][hook_name]["Scaled" if scale else "Unscaled"]
            df["Two Features (diff)"] = (df["YYY"] - df["NNN"]) - ((df["YNY"] - df["NNN"]) + (df["NYY"] - df["NNN"]) + (df["YYN"] - df["NNN"]))/2
            df["Single Features (diff)"] = (df["YYY"] - df["NNN"]) - ((df["YNN"] - df["NNN"]) + (df["NYN"] - df["NNN"]) + (df["NNY"] - df["NNN"]))
            df["Current Token (diff)"] = ((df["YYY"] - df["NYN"]) - ((df["YYN"] - df["NYN"]) + (df["NYY"] - df["NYN"])))
            df["Grouped Tokens (diff)"] = ((df["YYY"] - df["NNN"]) - ((df["YYN"] - df["NNN"]) + (df["NNY"] - df["NNN"])))
            df["Two Features (AND)"] = ((df["YYY"] - df["NNN"]) > ((df["YNY"] - df["NNN"]) + (df["NYY"] - df["NNN"]) + (df["YYN"] - df["NNN"]))/2) & (df["YYY"]>0)
            df["Single Features (AND)"] = ((df["YYY"] - df["NNN"]) > ((df["YNN"] - df["NNN"]) + (df["NYN"] - df["NNN"]) + (df["NNY"] - df["NNN"]))) & (df["YYY"]>0)
            df["Current Token (AND)"] = ((df["YYY"] - df["NYN"]) > ((df["YYN"] - df["NYN"]) + (df["NYY"] - df["NYN"]))) & (df["YYY"]>0)
            df["Grouped Tokens (AND)"] = ((df["YYY"] - df["NNN"]) > ((df["YYN"] - df["NNN"]) + (df["NNY"] - df["NNN"]))) & (df["YYY"]>0)
            df["Two Features (NEG AND)"] = ((df["YYY"] - df["NNN"]) < ((df["YNY"] - df["NNN"]) + (df["NYY"] - df["NNN"]) + (df["YYN"] - df["NNN"]))/2) & (df["NNN"]>0)
            df["Single Features (NEG AND)"] = ((df["YYY"] - df["NNN"]) < ((df["YNN"] - df["NNN"]) + (df["NYN"] - df["NNN"]) + (df["NNY"] - df["NNN"]))) & (df["NNN"]>0)
            df["Current Token (NEG AND)"] = ((df["YYY"] - df["NYN"]) < ((df["YYN"] - df["NYN"]) + (df["NYY"] - df["NYN"]))) & (df["NNN"]>0)
            df["Grouped Tokens (NEG AND)"] = ((df["YYY"] - df["NNN"]) < ((df["YYN"] - df["NNN"]) + (df["NNY"] - df["NNN"]))) & (df["NNN"]>0)
            df["Greater Than All"] = (df["YYY"] > df["NNN"]) & (df["YYY"] > df["YNN"]) & (df["YYY"] > df["NYN"]) & (df["YYY"] > df["NNY"]) & (df["YYY"] > df["YYN"]) & (df["YYY"] > df["NYY"]) & (df["YYY"] > df["YNY"])
            df["Smaller Than All"] = (df["YYY"] < df["NNN"]) & (df["YYY"] < df["YNN"]) & (df["YYY"] < df["NYN"]) & (df["YYY"] < df["NNY"]) & (df["YYY"] < df["YYN"]) & (df["YYY"] < df["NYY"]) & (df["YYY"] < df["YNY"])
            dfs[option][hook_name]["Scaled" if scale else "Unscaled"] = df

with open("data/and_neurons/activation_dfs.pkl", "wb") as f:
    pickle.dump(dfs, f)
# %%

for option in tqdm(options):
    prompts = all_prompts[option][:1000]

    # Ablation loss should be identical for all settings
    df = dfs[option]["hook_post"]["Unscaled"]

    original_loss, ablated_loss = haystack_utils.compute_mlp_loss(prompts, model, df, torch.LongTensor([i for i in range(model.cfg.d_mlp)]).cuda(), compute_original_loss=True, ablate_mode="YYN")
    print(original_loss, ablated_loss)

    ablated_losses = []
    for neuron in tqdm(range(model.cfg.d_mlp)):
        ablated_loss = haystack_utils.compute_mlp_loss(prompts, model, df, torch.LongTensor([neuron]).cuda(), ablate_mode="YYN")
        ablated_losses.append(ablated_loss)

    ablation_loss_increase = np.array(ablated_losses) - original_loss

    for hook_name in ["hook_pre", "hook_post"]:
        for scale in [True, False]:
            df = dfs[option][hook_name]["Scaled" if scale else "Unscaled"]
            df["AblationLossIncrease"] = ablation_loss_increase
            dfs[option][hook_name]["Scaled" if scale else "Unscaled"] = df

with open("data/and_neurons/activation_dfs.pkl", "wb") as f:
    pickle.dump(dfs, f)
# %%

# Ablation losses
all_losses = {}
for option in tqdm(options):
    all_losses[option] = {}
    prompts = all_prompts[option][:1000]
    for hook_name in ["hook_pre", "hook_post"]:
        all_losses[option][hook_name] = {}
        for scale in [True, False]:
            all_losses[option][hook_name]["Scaled" if scale else "Unscaled"] = {}
            df = dfs[option][hook_name]["Scaled" if scale else "Unscaled"]
            for include_mode in ["All Positive", "Greater Positive", "All Positive (Top 50)", "Greater Positive (Top 50)", "All Negative", "Smaller Negative", "All Negative (Top 50)", "Smaller Negative (Top 50)", "Positive and Negative (Top 25)"]:
                original_loss, all_ablated_loss = haystack_utils.compute_mlp_loss(prompts, model, df, torch.LongTensor([i for i in range(model.cfg.d_mlp)]), ablate_mode="YYN", compute_original_loss=True)
                all_losses[option][hook_name]["Scaled" if scale else "Unscaled"][include_mode] = {
                    "Original": original_loss,
                    "All Ablated": all_ablated_loss,
                }
                for feature_mode in ["Two Features", "Single Features", "Current Token", "Grouped Tokens"]:
                    if include_mode == "Greater Positive":
                        neurons = df[df[feature_mode + " (AND)"] & df["Greater Than All"]].index
                    elif include_mode == "All Positive":
                        neurons = df[df[feature_mode + " (AND)"]].index
                    elif include_mode == "Greater Positive (Top 50)":
                        neurons = haystack_utils.get_top_k_neurons(df, (df["YYY"]>0)&(df["Greater Than All"]), feature_mode + " (diff)", 50)
                    elif include_mode == "All Positive (Top 50)":
                        neurons = haystack_utils.get_top_k_neurons(df, (df["YYY"]>0), feature_mode + " (diff)", 50)
                    elif include_mode == "Smaller Negative":
                        neurons = df[df[feature_mode + " (NEG AND)"] & df["Smaller Than All"]].index
                    elif include_mode == "All Negative":
                        neurons = df[df[feature_mode + " (NEG AND)"]].index
                    elif include_mode == "Smaller Negative (Top 50)":
                        neurons = haystack_utils.get_top_k_neurons(df, (df["NNN"]>0)&(df["Smaller Than All"]), feature_mode + " (diff)", 50, ascending=True)
                    elif include_mode == "All Negative (Top 50)":
                        neurons = haystack_utils.get_top_k_neurons(df, (df["NNN"]>0), feature_mode + " (diff)", 50, ascending=True)
                    elif include_mode == "Positive and Negative (Top 25)":
                        neurons_top = haystack_utils.get_top_k_neurons(df, (df["NNN"]>0), feature_mode + " (diff)", 25, ascending=True)
                        neurons_bottom = haystack_utils.get_top_k_neurons(df, (df["YYY"]>0), feature_mode + " (diff)", 25)
                        neurons = np.concatenate([neurons_top, neurons_bottom])
                    else:
                        assert False, f"Invalid include mode: {include_mode}"
                    neurons = torch.LongTensor(neurons.tolist())

                    ablated_loss = haystack_utils.compute_mlp_loss(prompts, model, df, neurons, ablate_mode="YYN")
                    all_losses[option][hook_name]["Scaled" if scale else "Unscaled"][include_mode][feature_mode+f" (N={neurons.shape[0]})"] = ablated_loss

# %%
with open("data/and_neurons/ablation_losses.json", "w") as f:
    json.dump(all_losses, f, indent=4)
# %%

# Dataset prompts

# The random + ngram prompt
# %%
# %%
