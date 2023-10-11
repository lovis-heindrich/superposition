import random
import argparse
import pickle
import os
import gzip
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from einops import einops
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display, HTML
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

from neel_plotly import *
import haystack_utils
import probing_utils


def get_model(model_name: str, checkpoint: int) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(
        model_name,
        checkpoint_index=checkpoint,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return model


def process_data(model_name: str, output_dir: Path, image_dir: Path) -> None:
    model = get_model(model_name, 0)
    with gzip.open(
            output_dir.joinpath(model_name + "_checkpoint_features.pkl.gz"), "rb"
        ) as f:
        data = pickle.load(f)

    probe_df = data['probe']

    checkpoints = []
    top_probe = []
    for checkpoint in probe_df["Checkpoint"].unique():
        checkpoint_df = probe_df[probe_df["Checkpoint"] == checkpoint]
        top_probe.append(checkpoint_df["MCC"].max())
        checkpoints.append(checkpoint)
    fig = px.line(
        x=checkpoints,
        y=top_probe,
        title="Top Probe MCC by Checkpoint",
        width=800,
        height=400,
    )
    fig.write_image(image_dir.joinpath("top_mcc_by_checkpoint.png"))

    accurate_neurons = probe_df[
        (probe_df["MCC"] > 0.85)
        & (probe_df["MeanGermanActivation"] > probe_df["MeanNonGermanActivation"])
    ][["NeuronLabel", "MCC"]].copy()
    accurate_neurons = accurate_neurons.sort_values(by="MCC", ascending=False)
    print(
        len(accurate_neurons["NeuronLabel"].unique()),
        "neurons with an MCC > 0.85 for German text recognition at any point during training.",
    )

    good_neurons = accurate_neurons["NeuronLabel"].unique()[:50]

    # Melt the DataFrame
    probe_df_melt = probe_df[probe_df["NeuronLabel"].isin(good_neurons)].melt(id_vars=['Checkpoint'], var_name='NeuronLabel', value_vars="F1", value_name='F1 score')
    probe_df_melt['F1 score'] = pd.to_numeric(probe_df_melt['F1 score'], errors='coerce')

    # Calculate percentiles at each x-coordinate
    percentiles = [0.25, 0.5, 0.75]
    
    grouped = probe_df_melt.groupby('Checkpoint')['F1 score'].describe(percentiles=percentiles).reset_index()
    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['25%'], fill=None, mode='lines', line_color='rgba(0,100,80,0.2)', showlegend=False))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['75%'], fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line_color='rgba(0,100,80,0.2)', name="25th-75th percentile"))
    fig.add_trace(go.Scatter(x=grouped['Checkpoint'], y=grouped['50%'], mode='lines', line=dict(color='rgb(0,100,80)', width=2), name="Median"))

    fig.update_layout(title="F1 score of top neurons over time", xaxis_title="Checkpoint", yaxis_title="F1 score")

    fig.write_image(image_dir.joinpath("top_f1s_with_quartiles.png"))

    def get_mean_non_german(df, neuron, layer, checkpoint):
        label = f"C{checkpoint}L{layer}N{neuron}"
        df = df[df["Label"] == label]["MeanNonGermanActivation"].item()
        return df

    def get_mean_german(df, neuron, layer, checkpoint):
        label = f"C{checkpoint}L{layer}N{neuron}"
        df = df[df["Label"] == label]["MeanGermanActivation"].item()
        return df

    get_mean_non_german(probe_df, 669, 3, 140)

    layer_vals = np.random.randint(0, model.cfg.n_layers, good_neurons.size)
    neuron_vals = np.random.randint(0, model.cfg.d_mlp, good_neurons.size)
    random_neurons = probe_df[
        (probe_df["Layer"].isin(layer_vals)) & (probe_df["Neuron"].isin(neuron_vals))
    ]
    random_neurons = random_neurons["NeuronLabel"].unique()

    fig = px.line(
        probe_df[probe_df["NeuronLabel"].isin(good_neurons)],
        x="Checkpoint",
        y="MCC",
        color="NeuronLabel",
        title="Neurons with max MCC >= 0.85",
    )
    fig.write_image(image_dir.joinpath("high_mcc_neurons.png"))

    context_neuron_df = probe_df[probe_df["NeuronLabel"] == "L3N669"]
    fig = px.line(
        context_neuron_df,
        x="Checkpoint",
        y=["MeanGermanActivation", "MeanEnglishActivation"],
    )
    fig.write_image(image_dir.joinpath("mean_activations.png"))

    layer_ablation_df = data['layer_ablation']
    fig = px.line(
        layer_ablation_df.groupby(["Checkpoint", "Layer"]).mean().reset_index(),
        x="Checkpoint",
        y="LossDifference",
        color="Layer",
        title="Loss difference for zero-ablating MLP layers on German data",
        width=900,
    )
    fig.write_image(image_dir.joinpath("layer_ablation_losses.png"))



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

    save_path = os.path.join(args.output_dir, args.model)
    save_image_path = os.path.join(save_path, "images")

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_image_path, exist_ok=True)
    
    process_data(args.model, Path(save_path), Path(save_image_path))

