# %%
from pathlib import Path
from tqdm.auto import tqdm
from datasets import Dataset
from datasets import Dataset, load_dataset, load_from_disk
from transformer_lens import HookedTransformer
import sys
import os
import requests
import pandas as pd
from jaxtyping import Int
from torch import Tensor
import logging
from functools import lru_cache


def load_c4_tokens(data_path = "data/c4/data.hf", file_name = "data.hf") -> Int[Tensor, "n_samples seq_len"]:
    Path(data_path).mkdir(parents=True, exist_ok=True)
    file_path = f'{data_path}/{file_name}'
    if not os.path.exists(file_path):
        logging.info("Downloading c4 dataset")
        data = load_dataset("NeelNanda/pile-tokenized-10b", split="train")
        data.save_to_disk(file_path)
    
    data = load_dataset(data_path, data_files={'train': f'data-00000-of-00023.arrow', 'test': f'data-00003-of-00023.arrow'})
    data.set_format(type="torch", columns=["tokens"])
    data = data["tokens"]
    logging.info(f"Loaded c4 dataset with shape {data.shape}")
    return data


@lru_cache
def load_c4_validation_prompts(data_path = "/workspace/data/c4", file_name = "validation.parquet") -> list[str]:
    Path(data_path).mkdir(parents=True, exist_ok=True)
    file_path = f'{data_path}/{file_name}'
    if not os.path.isfile(file_path):
        print("Downloading c4 validation prompts")
        data = load_dataset("NeelNanda/pile-tokenized-10b", split="test")
        with open(file_path, "wb") as f:
            f.write(data)

    # Load Parquet into Pandas DataFrame
    df = pd.read_parquet(file_path, engine='pyarrow')
    prompts = df["text"].tolist()
    # Fix encoding issue for special characters like em-dash
    # prompts = [prompt.encode("windows-1252").decode("utf-8", errors="ignore") for prompt in prompts]
    logging.info(f"Loaded {len(prompts)} c4 validation prompts")
    return prompts

if __name__ == "__main__":
    data_path = "data/c4/data.hf"
    load_c4_tokens(data_path)
    load_c4_validation_prompts(data_path)