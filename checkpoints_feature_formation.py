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


def set_seeds():
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


def preload_models() -> int:
    """Preload models into cache so we can iterate over them quickly and return the model checkpoint count."""
    i = 0
    try:
        with tqdm(total=None) as pbar:
            while True:
                get_model(i)
                i += 1
                pbar.update(1)

    except IndexError:
        return i


def load_language_data() -> dict:
    """
    Returns: dictionary keyed by language code, containing 200 lines of each language included in the Europarl dataset.
    """
    lang_data = {}
    lang_data["en"] = haystack_utils.load_json_data("data/english_europarl.json")[:200]

    europarl_data_dir = Path("data/europarl/")
    for file in os.listdir(europarl_data_dir):
        if file.endswith(".txt"):
            lang = file.split("_")[0]
            lang_data[lang] = haystack_utils.load_txt_data(europarl_data_dir + file)

    for lang in lang_data.keys():
        print(lang, len(lang_data[lang]))
    return lang_data


def train_probe(
    positive_data: torch.Tensor, negative_data: torch.Tensor
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


def get_mlp_activations(
    prompts: list[str], layer: int, model: HookedTransformer
) -> torch.Tensor:
    act_label = f"blocks.{layer}.mlp.hook_post"

    acts = []
    for prompt in prompts:
        with model.hooks([(act_label, save_activation)]):
            model(prompt)
            act = model.hook_dict[act_label].ctx["activation"][:, 10:400, :]
        act = einops.rearrange(act, "batch pos n_neurons -> (batch pos) n_neurons")
        acts.append(act)
    acts = torch.concat(acts, dim=0)
    return acts[:k]


def zero_ablate_hook(value, hook):
    value[:, :, :] = 0
    return value


def get_layer_probe_performance(
    model: HookedTransformer,
    checkpoint: int,
    layer: int,
    german_data: np.array,
    non_german_data: np.array,
) -> pd.DataFrame:
    """Probe performance for each neuron."""

    german_activations = get_mlp_activations(german_data[:30], layer, model)[:10_000]
    non_german_activations = get_mlp_activations(non_german_data[:30], layer, model)[
        :10_000
    ]

    mean_german_activations = german_activations.mean(0).cpu().numpy()
    mean_non_german_activations = non_german_activations.mean(0).cpu().numpy()

    f1s = []
    mccs = []
    for neuron in range(model.cfg.d_mlp):
        f1, mcc = train_probe(
            german_activations[:, neuron].unsqueeze(-1),
            non_german_activations[:, neuron].unsqueeze(-1),
        )
        f1s.append(f1)
        mccs.append(mcc)

    checkpoint_neuron_labels = [
        f"C{checkpoint}L{layer}N{i}" for i in range(model.cfg.d_mlp)
    ]
    neuron_labels = [f"L{layer}N{i}" for i in range(model.cfg.d_mlp)]
    neuron_indices = [i for i in range(model.cfg.d_mlp)]

    layer_df = pd.DataFrame(
        {
            "Label": checkpoint_neuron_labels,
            "NeuronLabel": neuron_labels,
            "Neuron": neuron_indices,
            "F1": f1s,
            "MCC": mccs,
            "MeanGermanActivation": mean_german_activations,
            "MeanNonGermanActivation": mean_non_german_activations,
            "Checkpoint": checkpoint * len(checkpoint_neuron_labels),
            "Layer": layer * len(checkpoint_neuron_labels),
        }
    )
    return layer_df


def get_layer_ablation_loss(
    model: HookedTransformer, german_data: list, checkpoint: int, layer: int
):
    loss_data = []

    for prompt in german_data[:100]:
        loss = model(prompt, return_type="loss").item()
        with model.hooks([(f"blocks.{layer}.mlp.hook_post", zero_ablate_hook)]):
            ablated_loss = model(prompt, return_type="loss").item()
        loss_difference = ablated_loss - loss
        loss_data.append([checkpoint, layer, loss_difference, loss, ablated_loss])

    layer_df = pd.DataFrame(
        loss_data,
        columns=[
            "Checkpoint",
            "Layer",
            "LossDifference",
            "OriginalLoss",
            "AblatedLoss",
        ],
    )
    return layer_df


def get_language_losses(model: HookedTransformer, checkpoint: int, lang_data: dict) -> pd.DataFrame:
    data = []
    for lang in lang_data.keys():
        losses = []
        for prompt in lang_data[lang]:
            loss = model(prompt, return_type="loss").item()
            losses.append(loss)
        data.append([checkpoint, lang, np.mean(losses)])

    return pd.DataFrame(data, columns=["Checkpoint", "Language", "Loss"], index=[0])


def run_probe_analysis(
    model_name: str,
    num_checkpoints: int,
    lang_data: dict,
    output_dir: str,
) -> None:
    n_layers = get_model(model_name, 0).cfg.n_layers
    german_data = lang_data["de"]
    non_german_data = np.random.shuffle(
        np.concatenate([lang_data[lang] for lang in lang_data.keys() if lang != "de"])
    ).tolist()

    probe_dfs = []
    layer_ablation_dfs = []
    lang_loss_dfs = []

    with tqdm(total=checkpoints * n_layers) as pbar:
        for checkpoint in range(num_checkpoints):
            model = get_model(checkpoint)
            for layer in range(n_layers):
                partial_probe_df = get_layer_probe_performance(
                    model, checkpoint, layer, german_data, non_german_data
                )
                probe_dfs.append(partial_probe_df)

                partial_layer_ablation_df = get_layer_ablation_loss(
                    model, german_data, checkpoint, layer
                )
                layer_ablation_dfs.append(partial_layer_ablation_df)

                lang_loss_dfs.append(get_language_losses(model, checkpoint, lang_data))

                # Save progress to allow for checkpointing the analysis
                with open(
                    output_dir + model_name + "_checkpoint_features.pkl.gz", "wb"
                ) as f:
                    pickle.dump(
                        {
                            "probe": probe_dfs,
                            "layer_ablation": layer_ablation_dfs,
                            "lang_loss": lang_loss_dfs,
                        },
                        f,
                    )

                pbar.update(1)

    # Open the pickle file
    with open(output_dir + model_name + "_checkpoint_features.pkl.gz", "rb") as f:
        data = pickle.load(f)

    # Concatenate the dataframes
    data = {df_name: pd.concat(dfs) for dfs_name, dfs in data.items()}

    # Compress with gzip using high compression and save
    with gzip.open(
        output_dir + model_name + "_checkpoint_features.pkl.gz", "wb", compresslevel=9
    ) as f_out:
        pickle.dump(data, f_out)


def analyze_model_checkpoints(model_name: str, output_dir: str) -> None:
    set_seeds()

    # Load probe data
    lang_data = load_language_data()

    # Will take about 50GB of disk space for Pythia 70M models
    num_checkpoints = preload_models()

    run_probe_analysis(
        model_name, num_checkpoints, lang_data, output_dir
    )


def process_data(output_dir: str, model_name: str) -> None:
    def load_probe_analysis():
        with gzip.open(
            output_dir + model_name + "_checkpoint_neurons.pkl.gz", "rb"
        ) as f:
            data = pickle.load(f)
        return data

    probe_df = load_probe_analysis()

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
    # TODO Save plot

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

    fig = px.line(
        probe_df[
            probe_df["NeuronLabel"].isin(good_neurons)
            | probe_df["NeuronLabel"].isin(bad_neurons)
        ],
        x="Checkpoint",
        y="MCC",
        color="NeuronLabel",
        title="Neurons with max MCC >= 0.85",
    )
    fig.write_image(output_dir + "high_mcc_neurons.png")

    context_neuron_df = probe_df[probe_df["NeuronLabel"] == "L3N669"]
    fig = px.line(
        context_neuron_df,
        x="Checkpoint",
        y=["MeanGermanActivation", "MeanEnglishActivation"],
    )
    fig.write_image(output_dir + "mean_activations.png")

    with gzip.open(
        output_dir + model_name + "_checkpoint_layer_ablations.pkl.gz", "rb"
    ) as f:
        layer_ablation_df = pickle.load(f)

    fig = px.line(
        layer_ablation_df.groupby(["Checkpoint", "Layer"]).mean().reset_index(),
        x="Checkpoint",
        y="LossDifference",
        color="Layer",
        title="Loss difference for zero-ablating MLP layers on German data",
        width=900,
    )
    fig.write_image(output_dir + "layer_ablation_losses.png")


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

    analyze_model_checkpoints(args.model, args.output_dir)

    save_path = os.path.join(args.output_dir, args.model)
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{args.feature_dataset}_neurons.csv")
    results.to_csv(save_file, index=False)

    save_image = os.path.join(save_path, "images")
    os.makedirs(save_image, exist_ok=True)
    process_data(args.output_dir, args.model, save_image)
