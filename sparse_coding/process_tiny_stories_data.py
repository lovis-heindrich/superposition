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

def create_hf_dataset(seq_len=127):
    
    sys.path.append("../")  # Add the parent directory to the system path
    from utils.haystack_utils import get_device
    from utils.autoencoder_utils import batch_prompts

    model = HookedTransformer.from_pretrained(
        "tiny-stories-33M",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=get_device(),
    )

    dataset = load_dataset("roneneldan/TinyStories", split="train")
    prompts = []
    for i in tqdm(range(len(dataset))):
        text_prompt = dataset[i]["text"]
        # Fix encoding issue for special characters like em-dash
        # text_prompt = text_prompt.encode("windows-1252").decode("utf-8", errors="ignore")
        prompts.append(text_prompt)

    tokens = batch_prompts(prompts, model, seq_len=seq_len)
    #torch.save(tokens, f"{data_path}batched.pt")

    ds = Dataset.from_dict({"tokens": tokens})
    ds.push_to_hub("MechInterpResearch/tinystories_tokenized", private=True)

def load_tinystories_tokens(data_path = "data/tinystories", file_name = "data.hf", exclude_bos=False) -> Int[Tensor, "n_samples seq_len"]:
    Path(data_path).mkdir(parents=True, exist_ok=True)
    file_path = f'{data_path}/{file_name}'
    if not os.path.exists(file_path):
        logging.info("Downloading TinyStories dataset")
        data = load_dataset("MechInterpResearch/tinystories_tokenized", split="train")
        data.save_to_disk(file_path)
    
    data = load_from_disk(file_path)
    data.set_format(type="torch", columns=["tokens"])
    data = data["tokens"]
    if exclude_bos:
        data = data[:, 1:]
    logging.info(f"Loaded TinyStories dataset with shape {data.shape}")
    return  data

@lru_cache
def load_tinystories_validation_prompts(data_path = "/workspace/data/tinystories", file_name = "validation.parquet") -> list[str]:
    Path(data_path).mkdir(parents=True, exist_ok=True)
    file_path = f'{data_path}/{file_name}'
    if not os.path.isfile(file_path):
        print("Downloading TinyStories validation prompts")
        validation_data_path = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/validation-00000-of-00001-869c898b519ad725.parquet"
        response = requests.get(validation_data_path)
        with open(file_path, "wb") as f:
            f.write(response.content)

    # Load Parquet into Pandas DataFrame
    df = pd.read_parquet(file_path, engine='pyarrow')
    prompts = df["text"].tolist()
    # Fix encoding issue for special characters like em-dash
    # prompts = [prompt.encode("windows-1252").decode("utf-8", errors="ignore") for prompt in prompts]
    logging.info(f"Loaded {len(prompts)} TinyStories validation prompts")
    return prompts

if __name__ == "__main__":
    data_path = "data/tinystories"
    load_tinystories_tokens(data_path)
    load_tinystories_validation_prompts(data_path)


# %%
