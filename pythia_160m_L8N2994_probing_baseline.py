# %%
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import plotly.io as pio
import pandas as pd
import numpy as np
import plotly.express as px 
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import haystack_utils
import hook_utils
import plotting_utils
import probing_utils
from probing_utils import get_and_score_new_word_probe
from sklearn import preprocessing
from hook_utils import save_activation
from sklearn.utils import shuffle

pio.renderers.default = "notebook_connected+notebook"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)

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

LAYER, NEURON = 8, 2994

# %%
# IS_SPACE probes

def neuron_wise_activations(layer, num_examples=10000):
    neuron_space_activations = []
    neuron_non_space_activations = []
    hook_name = f'blocks.{layer}.mlp.hook_post'
    for prompt in german_data[:200]:
        tokens = model.to_tokens(prompt)[0]
        labels = probing_utils.get_new_word_labels(model, tokens)

        with model.hooks([(hook_name, save_activation)]):
            model(tokens)
        prompt_activations = model.hook_dict[hook_name].ctx['activation'][0, :-1]
        neuron_space_activations.append(prompt_activations[labels].cpu().numpy())
        neuron_non_space_activations.append(prompt_activations[~labels].cpu().numpy())

        # Early stopping
        neuron_space_activations_concat = np.concatenate(neuron_space_activations)
        neuron_non_space_activations_concat = np.concatenate(neuron_non_space_activations)
        if neuron_space_activations_concat.shape[0] >= num_examples and neuron_non_space_activations_concat.shape[0] >= num_examples:
            break
    neuron_space_activations = np.concatenate(neuron_space_activations)
    neuron_non_space_activations = np.concatenate(neuron_non_space_activations)
    print(neuron_space_activations.shape, neuron_non_space_activations.shape)
    return neuron_space_activations, neuron_non_space_activations
# %%
def neuron_wise_f1(neuron_pos_activations, neuron_neg_activations, layer, train_size=10000, test_size=10000, column_name=""):
    neuron_data = []
    for neuron in tqdm(range(model.cfg.d_mlp)):
        name = f"L{layer}N{neuron}"
        pos_activations = neuron_pos_activations[:(train_size+test_size)//2, neuron]
        neg_activations = neuron_neg_activations[:(train_size+test_size)//2, neuron]
        x = np.concatenate((pos_activations, neg_activations))
        x = x.reshape(-1, 1) # Reshape for sklearn
        scaler = preprocessing.StandardScaler()
        x = scaler.fit_transform(x)
        
        y = np.concatenate((np.full((train_size+test_size)//2, True), np.full((train_size+test_size)//2, False)))
        x, y = shuffle(x, y, random_state=42)
        
        probe = probing_utils.get_probe(x[:train_size], y[:train_size])
        f1, mcc = probing_utils.get_probe_score(probe, x[train_size:], y[train_size:])
        neuron_data.append([name, f1, mcc, pos_activations.mean(), neg_activations.mean(), layer, neuron])
        column_name = f" ({column_name})" if column_name else ""
    return pd.DataFrame(neuron_data, columns=["name", "f1"+column_name, "mcc"+column_name, "pos act"+column_name, "neg act"+column_name, "layer", "neuron"])

# def neuron_wise_f1(neuron_space_activations, neuron_non_space_activations, layer, train_size=10000, test_size=10000):
#     neuron_data = []
#     for neuron in tqdm(range(model.cfg.d_mlp)):
#         name = f"L{layer}N{neuron}"
#         space_activations = neuron_space_activations[:(train_size+test_size)//2, neuron]
#         non_space_activations = neuron_non_space_activations[:(train_size+test_size)//2, neuron]
#         x = np.concatenate((space_activations, non_space_activations))
#         x = x.reshape(-1, 1) # Reshape for sklearn
#         scaler = preprocessing.StandardScaler()
#         x = scaler.fit_transform(x)
        
#         y = np.concatenate((np.full((train_size+test_size)//2, True), np.full((train_size+test_size)//2, False)))
#         x, y = shuffle(x, y, random_state=42)
        
#         probe = probing_utils.get_probe(x[:train_size], y[:train_size])
#         f1, mcc = probing_utils.get_probe_score(probe, x[train_size:], y[train_size:])
#         neuron_data.append([name, f1, mcc, space_activations.mean(), non_space_activations.mean(), layer, neuron])
#     return pd.DataFrame(neuron_data, columns=["name", "f1", "mcc", "space_act", "non_space_act", "layer", "neuron"])
# %%
train_size=10000
test_size=10000
neuron_dfs = []
for layer in [8]:#range(9):
    neuron_space_activations, neuron_non_space_activations = neuron_wise_activations(layer, num_examples=(train_size+test_size)//2)
    neuron_df = neuron_wise_f1(neuron_space_activations, neuron_non_space_activations, layer, train_size=train_size, test_size=test_size, column_name="is_space")
    neuron_dfs.append(neuron_df)
full_df = pd.concat(neuron_dfs)

# %%
#full_df.to_csv("data/pythia_160m/space_probing_baseline.csv")
#%%
full_df = full_df.sort_values("f1", ascending=False)
full_df.head(10)
# %%

## GERMAN PROBES

def neuron_wise_german_activations(layer, num_examples=10000):
    german_activations = []
    english_activations = []
    hook_name = f'blocks.{layer}.mlp.hook_post'
    for prompt in german_data[:200]:
        tokens = model.to_tokens(prompt)[0]
        with model.hooks([(hook_name, save_activation)]):
            model(tokens)
        prompt_activations = model.hook_dict[hook_name].ctx['activation'][0, 10:]
        german_activations.append(prompt_activations.cpu().numpy())
        # Early stopping
        german_activations_concat = np.concatenate(german_activations)
        if german_activations_concat.shape[0] >= num_examples:
            break
    for prompt in english_data[:200]:
        tokens = model.to_tokens(prompt)[0]
        with model.hooks([(hook_name, save_activation)]):
            model(tokens)
        prompt_activations = model.hook_dict[hook_name].ctx['activation'][0, 10:]
        english_activations.append(prompt_activations.cpu().numpy())
        # Early stopping
        english_activations_concat = np.concatenate(english_activations)
        if english_activations_concat.shape[0] >= num_examples:
            break
    german_activations = np.concatenate(german_activations)
    english_activations = np.concatenate(english_activations)
    print(german_activations.shape, english_activations.shape)
    return german_activations, english_activations

def neuron_wise_f1(pos_activations, neg_activations, layer, train_size=10000, test_size=10000, column_name=""):
    neuron_data = []
    for neuron in tqdm(range(model.cfg.d_mlp)):
        name = f"L{layer}N{neuron}"
        pos_activations_activations = pos_activations[:(train_size+test_size)//2, neuron]
        neg_activations = neg_activations[:(train_size+test_size)//2, neuron]
        x = np.concatenate((pos_activations_activations, neg_activations))
        x = x.reshape(-1, 1) # Reshape for sklearn
        scaler = preprocessing.StandardScaler()
        x = scaler.fit_transform(x)
        
        y = np.concatenate((np.full((train_size+test_size)//2, True), np.full((train_size+test_size)//2, False)))
        x, y = shuffle(x, y, random_state=42)
        
        probe = probing_utils.get_probe(x[:train_size], y[:train_size])
        f1, mcc = probing_utils.get_probe_score(probe, x[train_size:], y[train_size:])
        neuron_data.append([name, f1, mcc, pos_activations_activations.mean(), neg_activations.mean(), layer, neuron])
        column_name = f" ({column_name})" if column_name else ""
    return pd.DataFrame(neuron_data, columns=["name", "f1"+column_name, "mcc"+column_name, "pos act"+column_name, "neg act"+column_name, "layer", "neuron"])

