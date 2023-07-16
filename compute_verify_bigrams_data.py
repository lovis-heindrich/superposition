
# %% 
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer, ActivationCache, utils, patching
from jaxtyping import Float, Int, Bool
from torch import Tensor
from tqdm.auto import tqdm
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display, clear_output
import json

pio.renderers.default = "notebook_connected"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)

from haystack_utils import get_mlp_activations
import haystack_utils

# %%
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device)

german_data = haystack_utils.load_json_data("data/german_europarl.json")[:200]
english_data = haystack_utils.load_json_data("data/english_europarl.json")[:200]


english_activations = {}
german_activations = {}
for layer in range(3, 4):
    english_activations[layer] = get_mlp_activations(english_data, layer, model, mean=False)
    german_activations[layer] = get_mlp_activations(german_data, layer, model, mean=False)

LAYER_TO_ABLATE = 3
NEURONS_TO_ABLATE = [669]
MEAN_ACTIVATION_ACTIVE = german_activations[LAYER_TO_ABLATE][:, NEURONS_TO_ABLATE].mean()
MEAN_ACTIVATION_INACTIVE = english_activations[LAYER_TO_ABLATE][:, NEURONS_TO_ABLATE].mean()

def deactivate_neurons_hook(value, hook):
    value[:, :, NEURONS_TO_ABLATE] = MEAN_ACTIVATION_INACTIVE
    return value
deactivate_neurons_fwd_hooks=[(f'blocks.{LAYER_TO_ABLATE}.mlp.hook_post', deactivate_neurons_hook)]

def activate_neurons_hook(value, hook):
    value[:, :, NEURONS_TO_ABLATE] = MEAN_ACTIVATION_ACTIVE
    return value
activate_neurons_fwd_hooks=[(f'blocks.{LAYER_TO_ABLATE}.mlp.hook_post', activate_neurons_hook)]

all_ignore, not_ignore = haystack_utils.get_weird_tokens(model, plot_norms=False)
# %%
# Get top common german tokens excluding punctuation
token_counts = torch.zeros(model.cfg.d_vocab).cuda()
for example in tqdm(german_data):
    tokens = model.to_tokens(example)
    for token in tokens[0]:
        token_counts[token.item()] += 1

punctuation = ["\n", ".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\", "\"", "'"]
leading_space_punctuation = [" " + char for char in punctuation]
punctuation_tokens = model.to_tokens(punctuation + leading_space_punctuation + [' –', " ", '  ', "<|endoftext|>"])[:, 1].flatten()
token_counts[punctuation_tokens] = 0
token_counts[all_ignore] = 0

top_counts, top_tokens = torch.topk(token_counts, 100)
print(model.to_str_tokens(top_tokens[:100]))

# %%

def get_random_selection(tensor, n=12):
    # Hacky replacement for np.random.choice
    return tensor[torch.randperm(len(tensor))[:n]]

def generate_random_prompts(end_string, n=50, length=12):
    # Generate a batch of random prompts ending with a specific ngram
    end_tokens = model.to_tokens(end_string).flatten()[1:]
    prompts = []
    for i in range(n):
        prompt = get_random_selection(top_tokens[:max(50, length)], n=length).cuda()
        prompt = torch.cat([prompt, end_tokens])
        prompts.append(prompt)
    prompts = torch.stack(prompts)
    return prompts

def replace_column(prompts: Int[Tensor, "n_prompts n_tokens"], token_index: int):
    # Replaces a specific token position in a batch of prompts with random common German tokens
    new_prompts = prompts.clone()
    random_tokens = get_random_selection(top_tokens[:max(50, prompts.shape[0])], n=prompts.shape[0]).cuda()
    new_prompts[:, token_index] = random_tokens
    return new_prompts 

def loss_analysis(prompts: Tensor, title=""):
    # Loss plot for a batch of prompts
    names = ["Original", "Ablated", "MLP5 path patched"]
    original_loss, ablated_loss, _, only_activated_loss = \
        haystack_utils.get_direct_effect(prompts, model, pos=-1,
                                        context_ablation_hooks=deactivate_neurons_fwd_hooks, 
                                        context_activation_hooks=activate_neurons_fwd_hooks, 
                                        )
    return original_loss.tolist(), ablated_loss.tolist(), only_activated_loss.tolist()

def loss_analysis_random_prompts(end_string, n=50, length=12, replace_columns: list[int] | None = None):
    # TODO fix for n>len(top_tokens)
    # Loss plot for a batch of random prompts ending with a specific ngram and optionally replacing specific tokens
    prompts = generate_random_prompts(end_string, n=n, length=length)
    title=f"Average last token loss on {length} random tokens ending in '{end_string}'"
    if replace_columns is not None:
        replaced_tokens = model.to_str_tokens(prompts[0, replace_columns])
        title += f" replacing {replaced_tokens}"
        for column in replace_columns:
            prompts = replace_column(prompts, column)
    
    return loss_analysis(prompts, title=title)
# %%
options = [" Vorschlägen", " häufig", " schließt", " beweglich"] 
option_lengths = [len(model.to_tokens(option, prepend_bos=False)[0]) for option in options]

# %%

all_loss_data = {}
for option in options:
    option_tokens = model.to_tokens(option, prepend_bos=False)[0]
    option_loss_data = {}

    original_loss, ablated_loss, only_activated_loss = loss_analysis_random_prompts(option, n=200, length=20)
    option_loss_data["None"] = {
        "Original": original_loss,
        "Ablated": ablated_loss,
        "MLP5 path patched": only_activated_loss,
    }

    for replace_pos in range(-len(option_tokens), -1, 1):
        replaced_token = model.to_str_tokens(option_tokens[replace_pos])
        original_loss, ablated_loss, only_activated_loss = loss_analysis_random_prompts(option, n=100, length=20, replace_columns=[replace_pos])
        option_loss_data[f"{replaced_token}"] = {
            "Original": original_loss,
            "Ablated": ablated_loss,
            "MLP5 path patched": only_activated_loss,
        }
    all_loss_data[option] = option_loss_data

# %%

with open('data/verify_bigrams/pos_loss_data.json', 'w') as f:
    json.dump(all_loss_data, f)
# %%
