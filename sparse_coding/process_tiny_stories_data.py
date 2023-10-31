# %%
import torch
from tqdm.auto import tqdm
import os
from datasets import Dataset
from datasets import Dataset, load_from_disk, concatenate_datasets, load_dataset
from transformer_lens import HookedTransformer
import sys 

sys.path.append('../')  # Add the parent directory to the system path
from utils.haystack_utils import get_device
from utils.autoencoder_utils import batch_prompts

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)


# %%

model = HookedTransformer.from_pretrained("tiny-stories-33M",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device="cuda")

dataset = load_dataset('roneneldan/TinyStories', split='train')
# %%
prompts = []
for i in tqdm(range(len(dataset))):
    prompts.append(dataset[i]['text'])

print(len(prompts))
# %%

tokens = batch_prompts(prompts, model, seq_len=127)
print(tokens.shape)
# %%
torch.save(tokens, "sparse_coding/data/tinystories/batched.pt")

# %%
ds = Dataset.from_dict({"tokens": tokens})
ds.push_to_hub("MechInterpResearch/tinystories_tokenized", private=True)
# %%
torch.numel(tokens)
# %%
