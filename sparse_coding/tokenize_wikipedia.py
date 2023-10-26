import tarfile
import re
import json
import os
import time
import sys
from itertools import islice
from typing import Iterable
import argparse

import torch
import einops
from datasets import load_dataset, Dataset
from transformer_lens import HookedTransformer
from tqdm import tqdm
import psutil
import ijson


sys.path.append("../")  # Add the parent directory to the system path
from utils.haystack_utils import get_device


# languages = ["bg", "cs", "da", "de", "el", "es", "en", "et", "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]
wikipedia_languages = ["de"]
datasets: dict[str, list[str]] = {lang: [] for lang in wikipedia_languages}


def add_to_dataset(data: Dataset, language: str, n_batches: int, batch_size: int, min_chars=500):
    n_lines = batch_size * n_batches
    lines_count = 0
    """Get lines of sufficient length and add to the global datasets dictionary"""
    for i, line in tqdm(enumerate(data)):
        if len(line["text"]) > min_chars:
            datasets[language].append(line["text"])
            lines_count += 1

        if lines_count == n_lines:
            break

        if psutil.virtual_memory().percent > 90:
            print(
                f"Memory overusage detected at {psutil.virtual_memory().percent}% and {i} lines, saving..."
            )
            with open(
                f"data/wikipedia/{language}_samples.json", "w", encoding="utf-8"
            ) as f:
                json.dump(datasets[language], f)
            print(f"\n{language}: {len(datasets[language])}")
            return

    print(
        f"No memory over-usage generating {sys.getsizeof(datasets)} bytes of data, saving..."
    )
    with open(f"data/wikipedia/{language}_samples.json", "w", encoding="utf-8") as f:
        json.dump(datasets[language], f)
    print(f"\n{language}: {len(datasets[language])}")


def chunked(iterable: list[str], n: int):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


def process_batch(data: list[str], model: HookedTransformer, seq_len: int):
    tensors = []
    for prompt in tqdm(data):
        tokens = model.to_tokens(prompt).cpu()
        tensors.append(tokens)

    batched_tensor = torch.cat(tensors, dim=1)
    batched_tensor = einops.rearrange(
        batched_tensor[
            :, : (batched_tensor.shape[1] - batched_tensor.shape[1] % seq_len)
        ],
        "batch (x seq_len) -> (batch x) seq_len",
        seq_len=seq_len,
    )
    return batched_tensor


def tokenize_dataset(language: str, n_batches: int, batch_size: int):
    seq_len = 500
    filename = f"data/wikipedia/{language}_samples.json"
    model = HookedTransformer.from_pretrained(
        "EleutherAI/pythia-70m",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=get_device(),
    )

    with open(filename, "rb") as f:
        items = ijson.items(f, "item")
        for i, batch in enumerate(chunked(items, batch_size)):
            if i == n_batches:
                break
            tensor = process_batch(batch, model, seq_len)
            torch.save(tensor, f"data/wikipedia/{language}_batched_{i}.pt")
            del tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        help="Number of batches to process",
        default=1,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of lines of data per batch",
        default=50_000,
    )
    args = parser.parse_args()

    os.makedirs("data/wikipedia", exist_ok=True)
    for language in wikipedia_languages:
        data: Dataset = load_dataset("wikipedia", f"20220301.{language}", split="train")
        add_to_dataset(data, language, args.n_batches, args.batch_size)
        tokenize_dataset(language, args.n_batches, args.batch_size)


# N = 600  # Estimated average string length
# GB = 1024 * 1024 * 1024  # Bytes in a GB

# # Estimate the number of strings
# num_strings = (8 * GB) // N
# num_tokens = num_strings * N // 2
# # Memory needed for token indices
# mem_for_tokens = num_tokens * 4  # in bytes
# mem_for_tokens_gb = mem_for_tokens / GB  # in GB

# # Memory needed for activations
# mem_for_activations = num_tokens * 2048 * 4  # in bytes
# mem_for_activations_gb = mem_for_activations / GB  # in GB

# print(f"Estimated memory for tokens: {mem_for_tokens_gb} GB")
# print(f"Estimated memory for activations: {mem_for_activations_gb} GB")
