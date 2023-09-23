### setup

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
import haystack_utils
from transformer_lens import utils
from fancy_einsum import einsum
import einops
import json
import ipywidgets as widgets
from IPython.display import display
from datasets import load_dataset
import random
import math
import random
import neel.utils as nutils
from neel_plotly import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import probing_utils
import pickle
from sklearn.metrics import matthews_corrcoef
import gzip
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotting_utils
import re

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

pio.renderers.default = "notebook_connected+notebook"
device = "cuda" if torch.cuda.is_available() else "cpu"
#torch.autograd.set_grad_enabled(False)
#torch.set_grad_enabled(False)

%reload_ext autoreload
%autoreload 2

# %%
def get_model(checkpoint: int) -> HookedTransformer:
    model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m",
        checkpoint_index=checkpoint,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=device)
    return model

NUM_CHECKPOINTS = 143
LAYER, NEURON = 3, 669
model = get_model(142)
german_data = haystack_utils.load_json_data("data/german_europarl.json")[:200]
english_data = haystack_utils.load_json_data("data/english_europarl.json")[:200]
all_ignore, _ = haystack_utils.get_weird_tokens(model, plot_norms=False)
common_tokens = haystack_utils.get_common_tokens(german_data, model, all_ignore, k=100)

def print_loss(model, prompt):
    '''Loss heatmap of tokens in prompt'''
    loss = model(prompt, return_type="loss", loss_per_token=True)[0]
    tokens = model.to_str_tokens(prompt)[1:]
    haystack_utils.print_strings_as_html(tokens, loss.tolist(), max_value=6)

def eval_loss(model, data):
    '''Mean of mean of token losses for each prompt'''
    losses = []
    for prompt in data:
        loss = model(prompt, return_type="loss")
        losses.append(loss.item())
    return np.mean(losses)

def eval_prompts(prompts, model, pos=-1):
    '''Mean loss at position in prompts'''
    loss = model(prompts, return_type="loss", loss_per_token=True)[:, pos].mean().item()
    return loss

def get_probe_performance(model, german_data, non_german_data, layer, neuron, plot=False):
    german_activations = haystack_utils.get_mlp_activations(german_data, layer, model, neurons=[neuron], mean=False)[:50000]
    non_german_activations = haystack_utils.get_mlp_activations(non_german_data, layer, model, neurons=[neuron], mean=False)[:50000]
    if plot:
        haystack_utils.two_histogram(german_activations.flatten(), non_german_activations.flatten(), "German", "NonGerman")
    return train_probe(german_activations, non_german_activations)

def train_probe(german_activations, non_german_activations):
    labels = np.concatenate([np.ones(len(german_activations)), np.zeros(len(non_german_activations))])
    activations = np.concatenate([german_activations.cpu().numpy(), non_german_activations.cpu().numpy()])
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    x_train, x_test, y_train, y_test = train_test_split(activations, labels, test_size=0.2, random_state=SEED)
    probe = probing_utils.get_probe(x_train, y_train, max_iter=2000)
    f1, mcc = probing_utils.get_probe_score(probe, x_test, y_test)
    return f1, mcc

def eval_checkpoint(checkpoint: int):
    model = get_model(checkpoint)
    german_loss = eval_loss(model, german_data)
    f1, mcc = get_probe_performance(model, german_data, non_german_data, LAYER, NEURON)
    return [checkpoint, german_loss, f1, mcc]

model = get_model(NUM_CHECKPOINTS-1)
non_german_activations = {}
for layer in range(3, 4):
    non_german_activations[layer] = haystack_utils.get_mlp_activations(non_german_data, layer, model, mean=False)

MEAN_ACTIVATION_INACTIVE = non_german_activations[LAYER][:, NEURON].mean()

def deactivate_neurons_hook(value, hook):
    value[:, :, NEURON] = MEAN_ACTIVATION_INACTIVE
    return value
deactivate_neurons_fwd_hooks=[(f'blocks.{LAYER}.mlp.hook_post', deactivate_neurons_hook)]

print(MEAN_ACTIVATION_INACTIVE)

# Will take about 50GB of disk space for Pythia 70M models
def preload_models(NUM_CHECKPOINTS: int):
    for i in tqdm(range(NUM_CHECKPOINTS)):
        get_model(i)
preload_models(NUM_CHECKPOINTS)

# %%
import os
from pathlib import Path

pile = {}
europarl_data_dir = Path("data/europarl/")
output_data_dir = Path("data/checkpoint/europarl/")

for file in os.listdir(europarl_data_dir):
    if file.endswith(".txt"):
        lang = file.split("_")[0]
        pile[lang] = haystack_utils.load_txt_data(europarl_data_dir + file)

for lang in pile.keys():
    print(lang, len(pile[lang]))

non_german_data = np.random.shuffle(np.concatenate([pile[lang] for lang in pile.keys() if lang != 'de'])).tolist()

### probe_df
# %%
# Probe performance for each neuron
def get_layer_probe_performance(model, checkpoint, layer):
    # english_activations = haystack_utils.get_mlp_activations(english_data[:30], layer, model, mean=False, disable_tqdm=True)[:10000]
    german_activations = haystack_utils.get_mlp_activations(german_data[:30], layer, model, mean=False, disable_tqdm=True)[:10000]
    non_german_activations = haystack_utils.get_mlp_activations(non_german_data[:30], layer, model, mean=False, disable_tqdm=True)[:10000]

    neuron_labels = [f'C{checkpoint}L{layer}N{i}' for i in range(model.cfg.d_mlp)]
    mean_non_german_activations = non_german_activations.mean(0).cpu().numpy()
    mean_german_activations = german_activations.mean(0).cpu().numpy()
    f1s = []
    mccs= []
    for neuron in range(model.cfg.d_mlp):
        f1, mcc = train_probe(german_activations[:, neuron].unsqueeze(-1), non_german_activations[:, neuron].unsqueeze(-1))
        f1s.append(f1)
        mccs.append(mcc)
    df = pd.DataFrame({"Label": neuron_labels, "Neuron": [i for i in range(model.cfg.d_mlp)], "F1": f1s, "MCC": mccs, "MeanGermanActivation": mean_german_activations, "MeanNonGermanActivation": mean_non_german_activations})
    df["Checkpoint"] = checkpoint
    df["Layer"] = layer
    return df

def run_probe_analysis(n_layers):
    dfs = []
    checkpoints = list(range(40)) + [40,50,60,70,80,90,100, 110, 120, 130, 140]
    with tqdm(total=len(checkpoints)*n_layers) as pbar:
        for checkpoint in checkpoints:
            model = get_model(checkpoint)
            for layer in range(n_layers):
                tmp_df = get_layer_probe_performance(model, checkpoint, layer)
                dfs.append(tmp_df)
                with open(output_data_dir + "layer_probe_performance_pile.pkl", "wb") as f:
                    pickle.dump(dfs, f)
                pbar.update(1)

    # Open the pickle file
    with open(output_data_dir + 'layer_probe_performance_pile.pkl', 'rb') as f:
        data = pickle.load(f)

    # Compress with gzip using high compression and save
    with gzip.open(output_data_dir + 'layer_probe_performance_pile.pkl.gz', 'wb', compresslevel=9) as f_out:
        pickle.dump(data, f_out)

# run_probe_analysis(model.cfg.n_layers)

def load_probe_analysis():
    with gzip.open(output_data_dir + 'layer_probe_performance_pile.pkl.gz', 'rb') as f:
        data = pickle.load(f)
    return data

# %%
dfs = load_probe_analysis()
probe_df = pd.concat(dfs)
probe_df["NeuronLabel"] = probe_df.apply(lambda row: f"L{row['Layer']}N{row['Neuron']}", axis=1)
probe_df.head()

# %%
checkpoints = []
top_probe = []
for checkpoint in probe_df["Checkpoint"].unique():
    tmp_df = probe_df[probe_df["Checkpoint"] == checkpoint]
    top_probe.append(tmp_df["MCC"].max())
    checkpoints.append(checkpoint)
px.line(x=checkpoints, y=top_probe, title="Top Probe MCC by Checkpoint", width=800, height=400)

# %%
neurons = probe_df[(probe_df["MCC"] > 0.85) & (probe_df["MeanGermanActivation"]>probe_df["MeanNonGermanActivation"])][["NeuronLabel", "MCC"]].copy()
neurons = neurons.sort_values(by="MCC", ascending=False)
print(len(neurons["NeuronLabel"].unique()))
good_neurons = neurons["NeuronLabel"].unique()[:50]

# %%
def get_mean_non_german(df, neuron, layer, checkpoint):
    label = f"C{checkpoint}L{layer}N{neuron}"
    df = df[df["Label"]==label]["MeanNonGermanActivation"].item()
    return df

def get_mean_german(df, neuron, layer, checkpoint):
    label = f"C{checkpoint}L{layer}N{neuron}"
    df = df[df["Label"]==label]["MeanGermanActivation"].item()
    return df

get_mean_non_german(probe_df, 669, 3, 140)

# %%
### ablation_df
# Ablation loss for top neurons

def run_ablation_analysis():
    ablation_data = []
    checkpoints = list(range(0, NUM_CHECKPOINTS, 10))
    print(checkpoints)
    with tqdm(total=len(checkpoints)*len(good_neurons)) as pbar:
        for checkpoint in checkpoints:
            model = get_model(checkpoint)
            for neuron_name in good_neurons:
                layer, neuron = neuron_name[1:].split("N")
                layer, neuron = int(layer), int(neuron)
                non_german_activations = get_mean_non_german(probe_df, neuron, layer, checkpoint)
                assert non_german_activations is not None
                def tmp_hook(value, hook):
                    value[:, :, neuron] = non_german_activations
                    return value
                tmp_hooks=[(f'blocks.{layer}.mlp.hook_post', tmp_hook)]
                original_loss = eval_loss(model, german_data)
                with model.hooks(tmp_hooks):
                    ablated_loss = eval_loss(model, german_data)
                ablation_data.append([neuron_name, checkpoint, original_loss, ablated_loss])
                pbar.update(1)

    ablation_df = pd.DataFrame(ablation_data, columns=["Label", "Checkpoint", "OriginalLoss", "AblatedLoss"])
    ablation_df["AblationIncrease"] = ablation_df["AblatedLoss"] - ablation_df["OriginalLoss"]
    ablation_df.to_csv("data/checkpoint_ablation_data.csv")

def load_ablation_analysis():
    ablation_df = pd.read_csv("data/checkpoint_ablation_data.csv")
    ablation_df["AblationIncrease"] = ablation_df["AblatedLoss"] - ablation_df["OriginalLoss"]
    return ablation_df

ablation_df = load_ablation_analysis()
ablation_df.head()

# %%


# Ablation loss for group of top neurons

def get_ablation_hook(neurons, layer, activations):
    def ablate_neurons_hook(value, hook):
        value[:, :, neurons] = activations
        return value
    return [(f'blocks.{layer}.mlp.hook_post', ablate_neurons_hook)]

def get_neuron_loss(checkpoint, neurons: list[str]):
    model = get_model(checkpoint)
    ablation_neurons = {l:[] for l in range(model.cfg.n_layers)}
    for neuron_name in neurons:
        layer, neuron = neuron_name[1:].split("N")
        layer, neuron = int(layer), int(neuron)
        ablation_neurons[layer].append(neuron)
    hooks = []
    for layer in range(model.cfg.n_layers):
        activations = []
        for neuron in ablation_neurons[layer]:
            label = f"C{checkpoint}L{layer}N{neuron}"
            activation = probe_df[probe_df["Label"]==label]["MeanEnglishActivation"].item()
            assert activation is not None
            activations.append(activation)
        activations = torch.tensor(activations).cuda()
        hooks.extend(get_ablation_hook(ablation_neurons[layer], layer, activations))
    original_loss = eval_loss(model, german_data)
    with model.hooks(hooks):
        ablated_loss = eval_loss(model, german_data)
    return original_loss, ablated_loss

# all_neuron_diffs = []
# for checkpoint in list(range(0, NUM_CHECKPOINTS, 10)):
#     original_loss, ablated_loss = get_neuron_loss(checkpoint, good_neurons)
#     diff = ablated_loss - original_loss
#     print(f"Checkpoint {checkpoint}: {original_loss} -> {ablated_loss}")
#     all_neuron_diffs.append(diff)

# all_neuron_df = pd.DataFrame({"Label": "Top 50", "Checkpoint": list(range(0, NUM_CHECKPOINTS, 10)), "AblationIncrease": all_neuron_diffs})
# ablation_df = pd.concat([ablation_df, all_neuron_df])
# ablation_df.head()

# %%

ablation_df.sort_values(by=["Checkpoint", "Label"], inplace=True)
px.line(ablation_df[ablation_df["Label"].isin(good_neurons)], x="Checkpoint", y="AblationIncrease", color="Label", title="Ablation Increase on German prompts", width=800)


# Ablation loss for selected neurons

def run_random_ablation_analysis(neurons: list[tuple[int, int]]):
    ablation_data = []
    checkpoints = list(range(0, NUM_CHECKPOINTS, 10))
    print(checkpoints)
    with tqdm(total=len(checkpoints)*len(good_neurons)) as pbar:
        for checkpoint in checkpoints:
            model = get_model(checkpoint)
            for layer, neuron in neurons:
                # layer, neuron = int(layer), int(neuron)
                # print(layer, neuron, checkpoint)
                # print(probe_df[probe_df["Label"]==f'C{checkpoint}L{layer}N{neuron}'])
                english_activations = get_mean_english(probe_df, neuron, layer, checkpoint)
                assert english_activations is not None
                def tmp_hook(value, hook):
                    value[:, :, neuron] = english_activations
                    return value
                tmp_hooks=[(f'blocks.{layer}.mlp.hook_post', tmp_hook)]
                original_loss = eval_loss(model, german_data)
                with model.hooks(tmp_hooks):
                    ablated_loss = eval_loss(model, german_data)
                ablation_data.append([f'L{layer}N{neuron}', checkpoint, original_loss, ablated_loss])
                pbar.update(1)

    random_ablation_df = pd.DataFrame(ablation_data, columns=["Label", "Checkpoint", "OriginalLoss", "AblatedLoss"])
    random_ablation_df["AblationIncrease"] = random_ablation_df["AblatedLoss"] - random_ablation_df["OriginalLoss"]
    random_ablation_df.to_csv("data/checkpoint_random_ablation_data.csv")

def load_random_ablation_analysis():
    random_ablation_df = pd.read_csv("data/checkpoint_random_ablation_data.csv")
    random_ablation_df["AblationIncrease"] = random_ablation_df["AblatedLoss"] - random_ablation_df["OriginalLoss"]
    return random_ablation_df

# %%

import numpy as np

# Pick as many random neurons as there are neurons with high MCC classifying German
layer_vals = np.random.randint(0, model.cfg.n_layers, good_neurons.size)
neuron_vals = np.random.randint(0, model.cfg.d_mlp, good_neurons.size)
random_neuron_indices = np.column_stack((layer_vals, neuron_vals))

# %%

random_ablation_df = load_random_ablation_analysis()
random_ablation_df.head()

# %%
random_neurons = probe_df[(probe_df['Layer'].isin(layer_vals)) & (probe_df['Neuron'].isin(neuron_vals))]
random_neurons = random_neurons["NeuronLabel"].unique()
# %%
random_ablation_df.sort_values(by=["Checkpoint", "Label"], inplace=True)
px.line(random_ablation_df[random_ablation_df["Label"].isin(random_neurons)], x="Checkpoint", y="AblationIncrease", color="Label", title="Ablation Increase of Random Neurons on German prompts", width=800)
# %%
px.line(probe_df[probe_df["NeuronLabel"].isin(good_neurons) | probe_df["NeuronLabel"].isin(bad_neurons)], x="Checkpoint", y="MCC", color="NeuronLabel", title="Neurons with max MCC >= 0.85")

# %%

context_neuron_df = probe_df[probe_df["NeuronLabel"]=="L3N669"]
px.line(context_neuron_df, x="Checkpoint", y=["MeanGermanActivation", "MeanEnglishActivation"])

# %%
### probe_df analysis over checkpoints

### L3N669 and vorschlagen losses analysis (context_neuron_eval) 

### DLA of L3N669 inputs over checkpoints - definitely a different script

### L3N669 losses on Pile datasets - could add the other languages here

### gradients analysis over whole model and good neurons (checkpoint_df)

### n-gram losses analysis (df)

### ablation analysis for L5N1712 on two languages (single_neuron_df_english, single_neuron_df_german)

### print activations for neurons on prompts

### layer ablation analysis (layer_df)

### ablating L3N669 on English data analysis (ablation_english_df)


### Tasks:
# - Move to this script
# - Add the other languages to the pile loss analysis
# - Add the other languages to the ablation analysis
# - New structure

For each checkpoint:
    layer ablation analysis (layer_df)


### L3N669 and vorschlagen losses analysis (context_neuron_eval) 



### L3N669 losses on Pile datasets - could add the other languages here

### gradients analysis over whole model and good neurons (checkpoint_df)

### n-gram losses analysis (df)

### ablation analysis for L5N1712 on two languages (single_neuron_df_english, single_neuron_df_german)

### print activations for neurons on prompts

### layer ablation analysis (layer_df)

### ablating L3N669 on English data analysis (ablation_english_df)

### DLA of L3N669 inputs over checkpoints - definitely a different script



