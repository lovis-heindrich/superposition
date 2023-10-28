# %%
import torch
from tqdm.auto import tqdm
import os
from datasets import Dataset
from datasets import Dataset, load_from_disk, concatenate_datasets, load_dataset

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)

# %%

# Directory to store shards
local_shard_dir = "data/hf_shards"
os.makedirs(local_shard_dir, exist_ok=True)

# %%
# Read PyTorch files and convert to Hugging Face dataset
for i in tqdm(range(46)):
    data_chunk = torch.load(f"data/wikipedia/de_batched_{i}.pt").tolist()
    ds_chunk = Dataset.from_dict({"tokens": data_chunk})
    # Save this chunk as a shard
    shard_path = f"{local_shard_dir}/shard_{i}"
    ds_chunk.save_to_disk(shard_path)

# %%
# Load all shards as a DatasetDict
shard_paths = [f"{local_shard_dir}/shard_{i}" for i in range(46)]
#dataset_dict = DatasetDict({f"shard_{i}": load_dataset(shard_path) for i, shard_path in enumerate(shard_paths)})
shard_datasets = [load_from_disk(shard_path) for shard_path in shard_paths]
concatenated_dataset = concatenate_datasets(shard_datasets)

# %%
concatenated_dataset.push_to_hub("lovish/german_wiki_tokenized", private=True)

# %%
from datasets import load_dataset
data = load_dataset("lovish/german_wiki_tokenized", split="train")
data.set_format(type="torch", columns=["tokens"])
all_tokens = data["tokens"]
all_tokens.shape
# %%

# Europarl
data = torch.load(f"data/europarl/de_batched.pt").tolist()
ds = Dataset.from_dict({"tokens": data})
ds.push_to_hub("lovish/german_europarl_tokenized", private=True)