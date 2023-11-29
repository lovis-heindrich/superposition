import json
import os
import sys
import argparse
import logging
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import plotly.io as pio
from jaxtyping import Int, Float
from torch import Tensor
import pandas as pd
sys.path.append('../')  # Add the parent directory to the system path
from process_tiny_stories_data import load_tinystories_validation_prompts
from pathlib import Path
from collections import Counter
import random
import plotly.express as px

pio.renderers.default = "notebook_connected"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)

logging.basicConfig(format='(%(levelname)s) %(asctime)s: %(message)s', level=logging.INFO, datefmt='%I:%M:%S')


import utils.haystack_utils as haystack_utils
from sparse_coding.train_autoencoder import AutoEncoder
from utils.autoencoder_utils import AutoEncoderConfig, load_encoder, get_activations, eval_direction_tokens_global


@torch.no_grad()
def get_acts(prompt: str, model: HookedTransformer, encoder: AutoEncoder, cfg: AutoEncoderConfig):
    _, cache = model.run_with_cache(prompt, names_filter=cfg.encoder_hook_point)
    acts = cache[cfg.encoder_hook_point].squeeze(0)
    _, _, mid_acts, _, _ = encoder(acts)
    return mid_acts



def print_top_examples(model: HookedTransformer, encoder: AutoEncoder, cfg: AutoEncoderConfig, prompts: list[str], activations: Float[Tensor, "n_prompts d_enc"], direction: int, n=5):
    top_idxs = activations[:, direction].argsort(descending=True)[:n].cpu().tolist()
    for prompt_index in top_idxs:
        prompt = prompts[prompt_index]
        prompt_tokens = model.to_str_tokens(model.to_tokens(prompt))
        acts = get_acts(prompt, model, encoder, cfg)
        direction_act = acts[:, direction].cpu().tolist()
        max_direction_act = max(direction_act)
        if max_direction_act > 0:
            haystack_utils.color_print_strings(prompt_tokens, direction_act, max_value=max_direction_act)
        else:
            print("No activations for this prompt.")

def load_model(model_name: str, device: str) -> HookedTransformer:
    return HookedTransformer.from_pretrained(
        model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=device,
    )

def get_direction_token_df(max_activations, prompts, model, encoder, encoder_cfg, percentage_threshold=0.5, save_path="/workspace/data/top_token_occurrences", force_recalculation=False):
    os.makedirs(save_path, exist_ok=True)
    file_name = f"{save_path}/{encoder_cfg.run_name}_direction_token_occurrences.csv"
    if os.path.exists(file_name) and not force_recalculation:
        direction_df = pd.read_csv(file_name)
    else:
        token_wise_activations = eval_direction_tokens_global(max_activations, prompts, model, encoder, encoder_cfg, percentage_threshold=percentage_threshold)
        total_occurrences = token_wise_activations.sum(1)
        max_occurrences = token_wise_activations.max(1)[0]
        max_occurring_token = token_wise_activations.argmax(1)

        direction_data = []
        for direction in tqdm(range(encoder.d_hidden)):
            total_occurrence = total_occurrences[direction].item()
            top_occurrence = max_occurrences[direction].item()
            top_token = model.to_single_str_token(max_occurring_token[direction].item())
            direction_data.append([direction, total_occurrence, top_token, top_occurrence])

        direction_df = pd.DataFrame(direction_data, columns=["Direction", "Total occurrences", "Top token", "Top token occurrences"])
        direction_df["Top token percent"] = direction_df["Top token occurrences"] / direction_df["Total occurrences"]
        direction_df = direction_df.dropna()
        direction_df.to_csv(file_name, index=False)
    return direction_df

def identify_low_density_directions(directions, max_activations: Float[Tensor, "n_prompts d_enc"], num_top_prompts_per_direction=5, num_most_common_prompts_considered=5):
    def get_top_prompt_indices(max_activations, direction, k):
        top_idxs = max_activations[:, direction].argsort(descending=True).cpu().tolist()[:k]
        # Filter by activation > 0 
        top_idxs = [idx for idx in top_idxs if max_activations[idx, direction] > 0]
        return top_idxs

    direction_top_indices = []
    for direction in directions:
        top_idxs = get_top_prompt_indices(max_activations, direction, k=num_top_prompts_per_direction)
        direction_top_indices.append(top_idxs)

    top_indices_counter = Counter([idx for top_idxs in direction_top_indices for idx in top_idxs])
    top_5_indices = [idx for idx, _ in top_indices_counter.most_common(num_most_common_prompts_considered)]

    clustered_direction = []
    for direction, top_indices in zip(directions, direction_top_indices):
        cluster_direction = False
        for top_index in top_indices:
            if top_index in top_5_indices:
                cluster_direction = True
        if cluster_direction:
            clustered_direction.append(direction)

    return clustered_direction

def identify_rare_directions(directions, max_activations: Float[Tensor, "n_prompts d_enc"], activation_threshold = 0, num_min_activating_prompts=1):
    num_directions = max_activations.shape[1]
    rare_directions = []
    for direction in directions:
        num_activating_prompts = (max_activations[:, direction] > activation_threshold).sum()
        if num_activating_prompts <= num_min_activating_prompts:
            rare_directions.append(direction)
    num_rare_directions = len(rare_directions)
    percent_rare_directions = num_rare_directions / num_directions
    return percent_rare_directions, rare_directions

def identify_single_token_directions(directions, direction_df, percent_threshold=0.8):
    single_token_directions = direction_df[direction_df["Top token percent"] > percent_threshold]["Direction"].tolist()
    single_token_directions = [int(direction) for direction in single_token_directions]
    single_token_directions = [direction for direction in single_token_directions if direction in directions]
    return single_token_directions


def main(model_name: str, run_name: str, label_file: str):
    if Path(label_file).is_file():
        overwrite = print("Model evaluation already exists. Overwrite? Y/n")
        if overwrite == 'n' or overwrite == 'N':
            return

    model = load_model(model_name, haystack_utils.get_device())
    encoder, cfg = load_encoder(run_name, model_name, model, save_path="/workspace")
    prompts = load_tinystories_validation_prompts()
    max_activations, _ = get_activations(encoder, cfg, run_name, prompts, model, save_activations=True)
    max_activations = max_activations.cpu()
    direction_df = get_direction_token_df(max_activations, prompts, model, encoder, cfg, percentage_threshold=0.5)

    directions = [i for i in range(encoder.d_hidden)]

    rare_directions = identify_rare_directions(directions, max_activations, num_min_activating_prompts=2, activation_threshold=0.1)
    percent_rare_directions = len(rare_directions) / encoder.d_hidden
    print(f"Percent rare directions: {percent_rare_directions}")

    directions = [i for i in directions if i not in rare_directions]

    clustered_directions = identify_low_density_directions(directions, max_activations)
    percent_low_density = len(clustered_directions) / encoder.d_hidden
    print(f"Percent low density directions: {percent_low_density}")
    
    directions = [i for i in directions if i not in clustered_directions]

    single_token_directions = identify_single_token_directions(directions, direction_df, percent_threshold=0.85)
    percent_single_token_directions = len(single_token_directions) / encoder.d_hidden
    print(f"Percent single token directions: {percent_single_token_directions}")

    directions = [i for i in directions if i not in single_token_directions]

    percent_good_directions = len(directions) / encoder.d_hidden

    num_manually_checked_prompts = 50
    labels = {label: [] for label in [1,2]}
    for direction in random.sample(directions, num_manually_checked_prompts):
        print_top_examples(model, encoder, cfg, prompts, max_activations, direction, n=10)

        label = input(f'Enter 1 for interpretable, and 2 for uninterpretable:')
        while label not in ['1', '2']:
            label = input('Invalid input.')
        labels[int(label)].append(direction)

    interpretable_directions = labels[1]
    uninterpretable_directions = labels[2]

    percent_interpretable = len(interpretable_directions) / num_manually_checked_prompts
    percent_uninterpretable = len(uninterpretable_directions) / num_manually_checked_prompts
    print(f"Percent good directions: {percent_good_directions}")
    print(f"Percent interpretable: {percent_interpretable}")
    print(f"Percent uninterpretable: {percent_uninterpretable}")

    res = {
        "percent_good_directions": percent_good_directions,
        "percent_low_density": percent_low_density,
        "percent_rare_directions": percent_rare_directions,
        "percent_single_token_directions": percent_single_token_directions,
        "percent_interpretable": percent_interpretable,
        "percent_uninterpretable": percent_uninterpretable,
        "interpretable_directions": interpretable_directions,
        "uninterpretable_directions": uninterpretable_directions,
        "good_directions": directions,
        "clustered_directions": clustered_directions,
        "rare_directions": rare_directions,
        "single_token_directions": single_token_directions,
    }

    with open(label_file, 'w') as f:
        json.dump(res, f)

    # Save plotly plot of all percentages as barplot
    # Creating a DataFrame from the given dictionary
    data = {
        "Metric": [
            "Low Density", 
            "Rare", 
            "Single Token",
            "Remaining",
            "Interpretable (of remaining)", 
            "Uninterpretable (of remaining)"
        ],
        "Percentage": [
            res["percent_low_density"], 
            res["percent_rare_directions"], 
            res["percent_single_token_directions"], 
            res["percent_good_directions"], 
            res["percent_interpretable"], 
            res["percent_uninterpretable"]
        ]
    }
    df = pd.DataFrame(data)

    # Creating a bar plot
    fig = px.bar(df, x='Metric', y='Percentage')
    fig.update_layout(title=f'{run_name} analysis', xaxis_title='Metric', yaxis_title='Percentage (%)')
    # save fig as png
    fig.write_image(f"./data/{args.model}/{run_name}_analysis.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tiny-stories-2L-33M"
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output",
        type=str,
    )
    args = parser.parse_args()

    os.makedirs(args.model, exist_ok=True)
    output = args.output or f"./data/{args.model}/{args.run}.json"
    os.makedirs(f"./data/{args.model}", exist_ok=True)
    
    main(args.model, args.run, output)
