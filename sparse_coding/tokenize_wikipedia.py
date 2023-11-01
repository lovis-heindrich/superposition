import json
import os
import sys
from itertools import islice
import argparse

import torch
from datasets import load_dataset, Dataset
from transformer_lens import HookedTransformer
from tqdm import tqdm
import psutil
import ijson


sys.path.append("../")  # Add the parent directory to the system path
from utils.haystack_utils import get_device
from utils.autoencoder_utils import batch_prompts


# languages = ["bg", "cs", "da", "de", "el", "es", "en", "et", "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]
wikipedia_languages = ["de"]
datasets: dict[str, list[str]] = {lang: [] for lang in wikipedia_languages}


def add_to_dataset(
    data: Dataset, language: str, n_batches: int, batch_size: int, min_chars=500
):
    save_filename = f"sparse_coding/data/wikipedia/{language}_samples.json"

    n_lines = batch_size * n_batches
    lines_count = 0
    """Get lines of sufficient length and add to the global datasets dictionary"""
    for i, line in tqdm(enumerate(data)):
        if len(line["text"]) > min_chars:
            datasets[language].append(line["text"])
            lines_count += 1

        # if lines_count == n_lines:
        #     break

        if psutil.virtual_memory().percent > 90:
            print(
                f"Memory overusage detected at {psutil.virtual_memory().percent}% and {i} lines, saving..."
            )
            with open(save_filename, "w", encoding="utf-8") as f:
                json.dump(datasets[language], f)
            print(f"\n{language}: {len(datasets[language])}")
            return

    with open(save_filename, "w", encoding="utf-8") as f:
        json.dump(datasets[language], f)
    print(
        f"Saved {len(datasets[language])} lines of {language} data, {os.path.getsize(save_filename)} bytes"
    )


def chunked(iterable: list[str], n: int):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


def tokenize_dataset(language: str, n_batches: int, batch_size: int):
    seq_len = 127
    filename = f"sparse_coding/data/wikipedia/{language}_samples.json"
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
            # if i == n_batches:
            #     break
            tensor = batch_prompts(batch, model, seq_len)
            torch.save(
                tensor, f"sparse_coding/data/wikipedia/{language}_batched_{i}.pt"
            )
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

    os.makedirs("sparse_coding/data/wikipedia", exist_ok=True)
    for language in wikipedia_languages:
        data: Dataset = load_dataset("wikipedia", f"20220301.{language}", split="train")
        add_to_dataset(data, language, args.n_batches, args.batch_size)
        tokenize_dataset(language, args.n_batches, args.batch_size)
