import random
import argparse
import json
import pickle
import os
import gzip
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from transformer_lens import HookedTransformer, utils
from datasets import load_dataset
from fancy_einsum import einsum
from tqdm.auto import tqdm
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display, HTML
from datasets import load_dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import plotly.express as px
import plotly.graph_objects as go

import neel.utils as nutils
from neel_plotly import *
import probing_utils
import haystack_utils


def setup():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    pio.renderers.default = "notebook_connected+notebook"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # torch.autograd.set_grad_enabled(False)
    # torch.set_grad_enabled(False)

    # %reload_ext autoreload
    # %autoreload 2

    NUM_CHECKPOINTS = 143


def get_model(checkpoint: int) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(
        "EleutherAI/pythia-70m",
        checkpoint_index=checkpoint,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=device,
    )
    return model


def train_probe(
    positive_data: np.array, negative_data: np.array
) -> tuple[float, float]:
    labels = np.concatenate([np.ones(len(positive_data)), np.zeros(len(negative_data))])
    data = np.concatenate([positive_data.cpu().numpy(), negative_data.cpu().numpy()])
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=SEED
    )
    probe = probing_utils.get_probe(x_train, y_train, max_iter=2000)
    f1, mcc = probing_utils.get_probe_score(probe, x_test, y_test)
    return f1, mcc


# Will take about 50GB of disk space for Pythia 70M models
def preload_models(NUM_CHECKPOINTS: int) -> None:
    for i in tqdm(range(NUM_CHECKPOINTS)):
        get_model(i)


def load_pile_data() -> dict:
    pile = {}
    europarl_data_dir = Path("data/europarl/")
    output_data_dir = Path("data/checkpoint/europarl/")

    for file in os.listdir(europarl_data_dir):
        if file.endswith(".txt"):
            lang = file.split("_")[0]
            pile[lang] = haystack_utils.load_txt_data(europarl_data_dir + file)

    for lang in pile.keys():
        print(lang, len(pile[lang]))
    return pile


def get_layer_probe_performance(
    model, checkpoint, layer, german_data, non_german_data
) -> pd.DataFrame:
    """Probe performance for each neuron"""

    # english_activations = haystack_utils.get_mlp_activations(english_data[:30], layer, model, mean=False, disable_tqdm=True)[:10000]
    german_activations = haystack_utils.get_mlp_activations(
        german_data[:30], layer, model, mean=False, disable_tqdm=True
    )[:10000]
    non_german_activations = haystack_utils.get_mlp_activations(
        non_german_data[:30], layer, model, mean=False, disable_tqdm=True
    )[:10000]

    neuron_labels = [f"C{checkpoint}L{layer}N{i}" for i in range(model.cfg.d_mlp)]
    mean_non_german_activations = non_german_activations.mean(0).cpu().numpy()
    mean_german_activations = german_activations.mean(0).cpu().numpy()
    f1s = []
    mccs = []
    for neuron in range(model.cfg.d_mlp):
        f1, mcc = train_probe(
            german_activations[:, neuron].unsqueeze(-1),
            non_german_activations[:, neuron].unsqueeze(-1),
        )
        f1s.append(f1)
        mccs.append(mcc)
    df = pd.DataFrame(
        {
            "Label": neuron_labels,
            "Neuron": [i for i in range(model.cfg.d_mlp)],
            "F1": f1s,
            "MCC": mccs,
            "MeanGermanActivation": mean_german_activations,
            "MeanNonGermanActivation": mean_non_german_activations,
        }
    )
    df["Checkpoint"] = checkpoint
    df["Layer"] = layer
    return df


def run_probe_analysis(model_name) -> None:
    n_layers = get_model(model_name, 0).cfg.n_layers

    dfs = []
    checkpoints = list(range(40)) + [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    with tqdm(total=len(checkpoints) * n_layers) as pbar:
        for checkpoint in checkpoints:
            model = get_model(checkpoint)
            for layer in range(n_layers):
                tmp_df = get_layer_probe_performance(model, checkpoint, layer)
                dfs.append(tmp_df)
                with open(
                    output_data_dir + "layer_probe_performance_pile.pkl", "wb"
                ) as f:
                    pickle.dump(dfs, f)
                pbar.update(1)

    # Open the pickle file
    with open(output_data_dir + "layer_probe_performance_pile.pkl", "rb") as f:
        data = pickle.load(f)

    # Compress with gzip using high compression and save
    with gzip.open(
        output_data_dir + "layer_probe_performance_pile.pkl.gz", "wb", compresslevel=9
    ) as f_out:
        pickle.dump(data, f_out)


def process_data() -> None:
    def load_probe_analysis():
        with gzip.open(
            output_data_dir + "layer_probe_performance_pile.pkl.gz", "rb"
        ) as f:
            data = pickle.load(f)
        return data

    dfs = load_probe_analysis()

    probe_df = pd.concat(dfs)
    probe_df["NeuronLabel"] = probe_df.apply(
        lambda row: f"L{row['Layer']}N{row['Neuron']}", axis=1
    )
    probe_df.head()

    checkpoints = []
    top_probe = []
    for checkpoint in probe_df["Checkpoint"].unique():
        tmp_df = probe_df[probe_df["Checkpoint"] == checkpoint]
        top_probe.append(tmp_df["MCC"].max())
        checkpoints.append(checkpoint)
    px.line(
        x=checkpoints,
        y=top_probe,
        title="Top Probe MCC by Checkpoint",
        width=800,
        height=400,
    )

    neurons = probe_df[
        (probe_df["MCC"] > 0.85)
        & (probe_df["MeanGermanActivation"] > probe_df["MeanNonGermanActivation"])
    ][["NeuronLabel", "MCC"]].copy()
    neurons = neurons.sort_values(by="MCC", ascending=False)
    print(len(neurons["NeuronLabel"].unique()))
    good_neurons = neurons["NeuronLabel"].unique()[:50]

    def get_mean_non_german(df, neuron, layer, checkpoint):
        label = f"C{checkpoint}L{layer}N{neuron}"
        df = df[df["Label"] == label]["MeanNonGermanActivation"].item()
        return df

    def get_mean_german(df, neuron, layer, checkpoint):
        label = f"C{checkpoint}L{layer}N{neuron}"
        df = df[df["Label"] == label]["MeanGermanActivation"].item()
        return df

    get_mean_non_german(probe_df, 669, 3, 140)

    random_neurons = probe_df[
        (probe_df["Layer"].isin(layer_vals)) & (probe_df["Neuron"].isin(neuron_vals))
    ]
    random_neurons = random_neurons["NeuronLabel"].unique()

    px.line(
        probe_df[
            probe_df["NeuronLabel"].isin(good_neurons)
            | probe_df["NeuronLabel"].isin(bad_neurons)
        ],
        x="Checkpoint",
        y="MCC",
        color="NeuronLabel",
        title="Neurons with max MCC >= 0.85",
    )

    context_neuron_df = probe_df[probe_df["NeuronLabel"] == "L3N669"]
    px.line(
        context_neuron_df,
        x="Checkpoint",
        y=["MeanGermanActivation", "MeanEnglishActivation"],
    )


### neurons' MCC analysis over checkpoints (probe_df)
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


# Script 1: checkpoints feature formation
# Input: --model_name="EleutherAI/pythia-70m"
# For each checkpoint:
### Don't include: neurons' MCC analysis over checkpoints - english vs. german (probe_df)
# neurons' MCC analysis over checkpoints - all langs vs. german (probe_df)
# Generate csv containing each neurons' final MCC

# layer ablation analysis (layer_df)
# German n-gram losses over checkpoints (df)

# Script 2: checkpoints contextual n-grams formation
# Input: --model_name="EleutherAI/pythia-70m" --context_neurons=[(3, 669)] --n_grams --dla=True --lang_losses=True --losses=True
# For each checkpoint:
# neuron and vorschlagen loss increases from neuron ablation (context_neuron_eval)
# neuron losses over other Pile datasets - could add the other Europarl languages here
# DLA of neuron input components over checkpoints
# If gradients specified:
# gradients analysis over whole model and good neurons (checkpoint_df)
# If prompt activations specified:
# print activations for neurons on prompts


# Not included (or generalize so that it is implicitly available):
### ablation analysis for L5N1712 on two languages (single_neuron_df_english, single_neuron_df_german)
### ablating L3N669 on English data analysis (ablation_english_df)


def analyze_model_checkpoints(model_name: str) -> None:
    setup()

    german_data = haystack_utils.load_json_data("data/german_europarl.json")[:200]
    english_data = haystack_utils.load_json_data("data/english_europarl.json")[:200]
    pile_data = load_pile_data()
    non_german_data = np.random.shuffle(
        np.concatenate([pile_data[lang] for lang in pile_data.keys() if lang != "de"])
    ).tolist()

    preload_models(NUM_CHECKPOINTS)

    run_probe_analysis(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="EleutherAI/pythia-70m",
        help="Name of model from TransformerLens",
    )
    parser.add_argument("--output_dir", default="feature_formation")

    args = parser.parse_args()

    results = analyze_model_checkpoints(args.model)

    save_path = os.path.join(args.output_dir, args.model)
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{args.feature_dataset}_neurons.csv")
    results.to_csv(save_file, index=False)
