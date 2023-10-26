import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import plotly.io as pio
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import wandb
import plotly.express as px
import pandas as pd
import torch.nn.init as init
import einops

pio.renderers.default = "notebook_connected"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)

import sys
sys.path.append('../')  # Add the parent directory to the system path
import utils.haystack_utils as haystack_utils

def batch_prompts(data: list[str], model: HookedTransformer, seq_len: int):
    tensors = []
    for prompt in tqdm(data):
        tokens = model.to_tokens(prompt).cpu()
        tensors.append(tokens)

    batched_tensor = torch.cat(tensors, dim=1)
    batched_tensor = einops.rearrange(batched_tensor[:, :(batched_tensor.shape[1]-batched_tensor.shape[1]%seq_len)], "batch (x seq_len) -> (batch x) seq_len", seq_len=seq_len)
    return batched_tensor

if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device)

    german_data = haystack_utils.load_json_data("data/europarl/de_samples.json")
    

    seq_len = 127
    german_tensor = batch_prompts(german_data, model, seq_len)
    batched_bos = torch.zeros(german_tensor.shape[0], 1, dtype=torch.long)
    batched_bos[:] = model.tokenizer.bos_token_id
    german_tensor = torch.cat([batched_bos, german_tensor], dim=1)
    german_tensor = german_tensor[torch.randperm(german_tensor.shape[0])]
    german_tensor = german_tensor.to(torch.int32)
    torch.save(german_tensor, "data/europarl/de_batched.pt")
    del german_tensor, german_data

    # english_data = haystack_utils.load_json_data("data/europarl/en_samples.json")
    # english_tensor = batch_prompts(english_data, model, seq_len)
    # torch.save(english_tensor, "data/europarl/en_batched.pt")