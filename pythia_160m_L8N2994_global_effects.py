#%%
import torch
from transformer_lens import HookedTransformer
from jaxtyping import Float, Int, Bool
from torch import Tensor
from tqdm.auto import tqdm
import plotly.io as pio
import haystack_utils
import pandas as pd
import plotly.express as px
import scipy.stats as stats

pio.renderers.default = "notebook_connected+notebook"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)



%reload_ext autoreload
%autoreload 2

model = HookedTransformer.from_pretrained("EleutherAI/pythia-160m",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device)

german_data = haystack_utils.load_json_data("data/german_europarl.json")[:200]
english_data = haystack_utils.load_json_data("data/english_europarl.json")[:200]
all_ignore, _ = haystack_utils.get_weird_tokens(model, plot_norms=False)
common_tokens = haystack_utils.get_common_tokens(german_data, model, all_ignore, k=100)

LAYER, NEURON = 8, 2994
neuron_activations = haystack_utils.get_mlp_activations(german_data, LAYER, model, neurons=torch.LongTensor([NEURON]), mean=False).flatten()

def get_next_token_punctuation_mask(tokens: torch.LongTensor) -> torch.BoolTensor:
    next_token_punctuation_mask = torch.zeros_like(tokens, dtype=torch.bool)
    token_strs = model.to_str_tokens(tokens)
    for i in range(tokens.shape[0] - 1):
        next_token_str = token_strs[i + 1]
        next_is_space = next_token_str[0] in [" ", ",", ".", ":", ";", "!", "?"]
        next_token_punctuation_mask[i] = next_is_space
    return next_token_punctuation_mask


# %%

def get_space_tokens():
    space_tokens = []
    non_space_tokens = []
    ignore_tokens = all_ignore.flatten().tolist()
    for i in range(model.cfg.d_vocab):
        string_token = model.to_single_str_token(i)
        if (not string_token) or (i in ignore_tokens):
            continue
        elif string_token[0] in [" ", ",", ".", ":", ";", "!", "?"]:
            space_tokens.append(i)
        else:
            non_space_tokens.append(i)
    return torch.tensor(space_tokens), torch.tensor(non_space_tokens)
# %%
# Get "is space" direction in embeddings
space_tokens, non_space_tokens = get_space_tokens()

# %% 
# def space_tokens_diff(space_tokens, non_space_tokens, original_logits, ablated_logits):
#     original_logit_diffs = original_logits[:, space_tokens].norm(dim=-1) - original_logits[:, non_space_tokens].norm(dim=-1)
#     ablated_logits_diffs = ablated_logits[:, space_tokens].norm(dim=-1) - ablated_logits[:, non_space_tokens].norm(dim=-1)
#     print(ablated_logits_diffs, original_logit_diffs)
#     return (ablated_logits_diffs - original_logit_diffs).mean().item()

def space_tokens_diff(space_tokens, non_space_tokens, original_probs, ablated_probs):
    space_diff = (ablated_probs[:, space_tokens] - original_probs[:, space_tokens]).mean()
    non_space_diff = (ablated_probs[:, non_space_tokens] - original_probs[:, non_space_tokens]).mean()
    return space_diff.item(), non_space_diff.item()

def snap_pos_to_peak_1(value, hook):
    '''Doesn't snap disabled and ambiguous activations'''
    value[:, :, NEURON] = 2.5
    return value

def snap_pos_to_peak_2(value, hook):
    '''Doesn't snap disabled and ambiguous activations'''
    value[:, :, NEURON] = 6.5
    return value

snap_pos_to_peak_1_hook = [(f'blocks.{LAYER}.mlp.hook_post', snap_pos_to_peak_1)]
snap_pos_to_peak_2_hook = [(f'blocks.{LAYER}.mlp.hook_post', snap_pos_to_peak_2)]

for prompt in german_data:
    with model.hooks(snap_pos_to_peak_1_hook):
        peak_1_prob = model(prompt, return_type="logits").softmax(dim=-1)
    with model.hooks(snap_pos_to_peak_2_hook):
        peak_2_prob = model(prompt, return_type="logits").softmax(dim=-1)
    print(space_tokens_diff(space_tokens, non_space_tokens, peak_1_prob[0], peak_2_prob[0]))
    break
# %%
