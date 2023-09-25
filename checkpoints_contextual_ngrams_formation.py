import random
import argparse
import pickle
import os
import gzip
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import torch
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display, HTML
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import plotly.express as px
from nltk import ngrams

from neel_plotly import *
import haystack_utils
import probing_utils


SEED = 42

def set_seeds():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    pio.renderers.default = "notebook_connected+notebook"


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


def preload_models(model_name: str) -> int:
    """Preload models into cache so we can iterate over them quickly and return the model checkpoint count."""
    i = 0
    try:
        with tqdm(total=None) as pbar:
            while True:
                get_model(model_name, i)
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
            lang_data[lang] = haystack_utils.load_txt_data(europarl_data_dir.joinpath(file))

    for lang in lang_data.keys():
        print(lang, len(lang_data[lang]))
    return lang_data


def eval_prompts(prompts, model, pos=-1):
    '''Mean loss at position in prompts'''
    loss = model(prompts, return_type="loss", loss_per_token=True)[:, pos].mean().item()
    return loss


def deactivate_neurons_hook(value, hook):
    value[:, :, NEURON] = MEAN_ACTIVATION_INACTIVE
    return value
deactivate_neurons_fwd_hooks=[(f'blocks.{LAYER}.mlp.hook_post', deactivate_neurons_hook)]


def get_ngram_losses(
    model: HookedTransformer,
    checkpoint: int,
    ngrams: list[str],
    common_tokens: list[str],
) -> pd.DataFrame:
    data = []
    for ngram in ngrams:
        prompts = haystack_utils.generate_random_prompts(
            ngram, model, common_tokens, 100, 20
        )
        loss = eval_prompts(prompts, model)
        with model.hooks(deactivate_neurons_fwd_hooks):
            ablated_loss = eval_prompts(prompts, model)
        data.append([loss, ablated_loss, ablated_loss - loss, checkpoint, ngram])

    df = pd.DataFrame(
        data,
        columns=["OriginalLoss", "AblatedLoss", "LossIncrease", "Checkpoint", "Ngram"],
    )
    return df


def get_common_ngrams(
    model: HookedTransformer, prompts: list[str], n: int, top_k=100
) -> list[str]:
    """
    n: n-gram length
    top_k: number of n-grams to return

    Returns: List of most common n-grams in prompts sorted by frequency
    """
    all_ngrams = []
    for prompt in tqdm(prompts):
        str_tokens = model.to_str_tokens(prompt)
        all_ngrams.extend(ngrams(str_tokens, n))
    # Filter n-grams which contain punctuation
    all_ngrams = [
        x
        for x in all_ngrams
        if all(
            [
                y.strip() not in ["\n", "-", "(", ")", ".", ",", ";", "!", "?", ""]
                for y in x
            ]
        )
    ]
    return Counter(all_ngrams).most_common(top_k)


def eval_loss(model, data, mean=True):
    '''Mean of mean of token losses for each prompt'''
    losses = []
    for prompt in data:
        loss = model(prompt, return_type="loss")
        losses.append(loss.item())
    if mean:
        return np.mean(losses)
    return losses

    
def eval_checkpoint(model: HookedTransformer, german_data: list[str], non_german_data: list[str], checkpoint: int, layer: int, neuron: int):
    model = get_model(checkpoint)
    german_loss = eval_loss(model, german_data)
    f1, mcc = get_probe_performance(model, german_data, non_german_data, layer, neuron)
    return [checkpoint, german_loss, f1, mcc]


def analyze_contextual_ngrams(
        model_name: str, 
        neurons: list[tuple[int, int]], 
        ngrams: bool, 
        dla: bool, 
        lang_losses: bool, 
        save_path: Path
    ):
    set_seeds()
    num_checkpoints = preload_models(model_name)
    lang_data = load_language_data()
    german_data = lang_data["de"]

    temp_model = get_model(model_name, 0)
    n_layers = temp_model.cfg.n_layers

    all_ignore, _ = haystack_utils.get_weird_tokens(model, plot_norms=False)    

    common_tokens = haystack_utils.get_common_tokens(
        german_data, model, all_ignore, k=100
    )
    top_trigrams = get_common_ngrams(model, lang_data["de"], 3, 200)
    random_trigram_indices = np.random.choice(
        range(len(top_trigrams)), 20, replace=False
    )
    random_trigrams = ["".join(top_trigrams[i][0]) for i in random_trigram_indices]

    ngram_loss_dfs = []
    context_neuron_eval_dfs = []
    with tqdm(total=num_checkpoints * n_layers) as pbar:
        for checkpoint in range(num_checkpoints):
            model = get_model(checkpoint)
            for layer in range(n_layers):
                ngram_loss_dfs.append(
                    get_ngram_losses(model, checkpoint, random_trigrams, common_tokens)
                )
                context_neuron_eval_dfs.append()
                pbar.update(1)


def tuple_list_type(value):
    return [tuple(map(int, t.split(','))) for t in value.split(' ')]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="EleutherAI/pythia-70m",
        help="Name of model from TransformerLens",
    )
    parser.add_argument("--output_dir", default="contextual_ngrams_formation")
    parser.add_argument("--neurons", default=[(3, 669)], type=tuple_list_type, 
                        help="Space delimited list of neurons to analyze. \
                        Each neuron is a comma delimited tuple of layer_index,neuron_index.")
    parser.add_argument("--ngrams", default=True)
    parser.add_argument("--dla", default=True)
    parser.add_argument("--lang_losses", default=True)

    args = parser.parse_args()

    save_path = os.path.join(args.output_dir, args.model)
    os.makedirs(save_path, exist_ok=True)
    
    analyze_contextual_ngrams(args.model, args.neurons, args.ngrams, args.dla, args.lang_losses, Path(save_path))

