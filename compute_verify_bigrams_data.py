
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
    indices = torch.randint(0, len(tensor), (n,))
    return tensor[indices]

def generate_random_prompts(end_string, n=50, length=12):
    # Generate a batch of random prompts ending with a specific ngram
    end_tokens = model.to_tokens(end_string).flatten()[1:]
    prompts = []
    for i in range(n):
        prompt = get_random_selection(top_tokens[:50], n=length).cuda()
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
        replaced_token = model.to_str_tokens(option_tokens[replace_pos])[0]
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
# %% 

individual_neuron_loss_diffs = {}

for option in options:

    prompts = generate_random_prompts(option, n=100, length=20)

    with model.hooks(deactivate_neurons_fwd_hooks):
        _, ablated_cache = model.run_with_cache(prompts)

    def get_ablate_neurons_hook(neuron: int | list[int], ablated_cache, layer=5):
        def ablate_neurons_hook(value, hook):
            value[:, :, neuron] = ablated_cache[f'blocks.{layer}.mlp.hook_post'][:, :, neuron]
            return value
        return [(f'blocks.{layer}.mlp.hook_post', ablate_neurons_hook)]

    diffs = torch.zeros(2048, prompts.shape[0])
    # Loss with path patched MLP5 neurons
    _, _, _, baseline_loss = haystack_utils.get_direct_effect(prompts, model, pos=-1, context_ablation_hooks=deactivate_neurons_fwd_hooks, context_activation_hooks=activate_neurons_fwd_hooks)
    for neuron in tqdm(range(2048)):
        ablate_single_neuron_hook = get_ablate_neurons_hook(neuron, ablated_cache)
        # Loss with path patched MLP5 neurons but a single neuron changed back to original ablated value
        _, _, _, only_activated_loss = haystack_utils.get_direct_effect(prompts, model, pos=-1, context_ablation_hooks=deactivate_neurons_fwd_hooks, context_activation_hooks=activate_neurons_fwd_hooks+ablate_single_neuron_hook)
        diffs[neuron] = only_activated_loss - baseline_loss

    individual_neuron_loss_diffs[option] = diffs.mean(1).tolist()
    # sorted_means, indices = torch.sort(diffs.mean(1))
    # sorted_means = sorted_means.tolist()
    # haystack_utils.line(sorted_means, xlabel="Sorted neurons", ylabel="Loss change", title="Loss change from ablating MLP5 neuron") # xticks=indices
# %%
with open('data/verify_bigrams/neuron_loss_diffs.json', 'w') as f:
    json.dump(individual_neuron_loss_diffs, f)

# %%

top_bottom_neuron_losses = {}

for option in tqdm(options):
    prompts = generate_random_prompts(option, n=200, length=20)

    diffs = individual_neuron_loss_diffs[option]
    sorted_means, indices = torch.sort(torch.tensor(diffs))
    sorted_means = sorted_means.tolist()

    top_bottom_neuron_losses[option] = {}

    for num_neurons in range(1, 26):
        # Check loss change when ablating top / bottom neurons
        top_neurons_count = num_neurons
        top_neurons = indices[-top_neurons_count:]
        bottom_neurons = indices[:top_neurons_count]

        with model.hooks(deactivate_neurons_fwd_hooks):
            ablated_loss, ablated_cache = model.run_with_cache(prompts, return_type="loss")

        ablate_top_neurons_hook = get_ablate_neurons_hook(top_neurons, ablated_cache)
        ablate_bottom_neurons_hook = get_ablate_neurons_hook(bottom_neurons, ablated_cache)

        original_loss, ablated_loss, _, all_MLP5_loss = haystack_utils.get_direct_effect(prompts, model, pos=-1, context_ablation_hooks=deactivate_neurons_fwd_hooks, context_activation_hooks=activate_neurons_fwd_hooks)
        _, _, _, top_MLP5_ablated_loss = haystack_utils.get_direct_effect(prompts, model, pos=-1, context_ablation_hooks=deactivate_neurons_fwd_hooks, context_activation_hooks=activate_neurons_fwd_hooks+ablate_top_neurons_hook)
        _, _, _, bottom_MLP5_ablated_loss = haystack_utils.get_direct_effect(prompts, model, pos=-1, context_ablation_hooks=deactivate_neurons_fwd_hooks, context_activation_hooks=activate_neurons_fwd_hooks+ablate_bottom_neurons_hook)

        names = ["Original", "Ablated", "MLP5 path patched", f"MLP5 path patched + Top {top_neurons_count} MLP5 neurons ablated", f"MLP5 path patched + Bottom {top_neurons_count} MLP5 neurons ablated"]
        short_names = ["Original", "Ablated", "MLP5 path patched", f"Top MLP5 removed", f"Bottom MLP5 removed"]

        top_bottom_neuron_losses[option][num_neurons] = {
            "Original": original_loss.tolist(),
            "Ablated": ablated_loss.tolist(),
            "MLP5 path patched": all_MLP5_loss.tolist(),
            f"Top MLP5 removed": top_MLP5_ablated_loss.tolist(),
            f"Bottom MLP5 removed": bottom_MLP5_ablated_loss.tolist()
        }

# %%
with open('data/verify_bigrams/neuron_loss_data.json', 'w') as f:
    json.dump(top_bottom_neuron_losses, f)
# %%
def get_top_difference_neurons(baseline_logprobs, ablated_logprops, positive=True, logprob_threshold=-7, k=50):
    neuron_logprob_difference = (baseline_logprobs - ablated_logprops).mean(0)
    neuron_logprob_difference[baseline_logprobs.mean(0) < logprob_threshold] = 0
    if positive:
        non_zero_count = (neuron_logprob_difference > 0).sum()
    else:
        non_zero_count = (neuron_logprob_difference < 0).sum()
    top_logprob_difference, top_neurons = haystack_utils.top_k_with_exclude(neuron_logprob_difference, min(non_zero_count, k), all_ignore, largest=positive)
    return top_logprob_difference, top_neurons

summed_neuron_logprob_boosts = {}
# Compute summed neuron boosts / deboosts
for option in options: 
    summed_neuron_logprob_boosts[option] = {}
    prompts = generate_random_prompts(option, n=200, length=20)

    diffs = individual_neuron_loss_diffs[option]
    sorted_means, indices = torch.sort(torch.tensor(diffs))
    sorted_means = sorted_means.tolist()

    for num_neurons in tqdm(range(1, 26)):
        
        summed_neuron_logprob_boosts[option][num_neurons] = {}
        summed_neuron_logprob_boosts[option][num_neurons]["Top"] = {}
        summed_neuron_logprob_boosts[option][num_neurons]["Bottom"] = {}

        top_neurons = indices[-num_neurons:]
        bottom_neurons = indices[:num_neurons]

        with model.hooks(deactivate_neurons_fwd_hooks):
            ablated_logits, ablated_cache = model.run_with_cache(prompts)

        ablate_top_neurons_hook = get_ablate_neurons_hook(top_neurons, ablated_cache)
        ablate_bottom_neurons_hook = get_ablate_neurons_hook(bottom_neurons, ablated_cache)

        original_logits, ablated_logprobs, _, all_MLP5_logprobs = haystack_utils.get_direct_effect(prompts, model, pos=-2, context_ablation_hooks=deactivate_neurons_fwd_hooks, context_activation_hooks=activate_neurons_fwd_hooks, return_type='logprobs')
        _, _, _, top_MLP5_ablated_logprobs = haystack_utils.get_direct_effect(prompts, model, pos=-2, context_ablation_hooks=deactivate_neurons_fwd_hooks, context_activation_hooks=activate_neurons_fwd_hooks+ablate_top_neurons_hook, return_type='logprobs')
        _, _, _, bottom_MLP5_ablated_logprobs = haystack_utils.get_direct_effect(prompts, model, pos=-2, context_ablation_hooks=deactivate_neurons_fwd_hooks, context_activation_hooks=activate_neurons_fwd_hooks+ablate_bottom_neurons_hook, return_type='logprobs')
            
        # Boosted tokens
        bottom_neuron_pos_difference_logprobs, bottom_pos_indices = get_top_difference_neurons(all_MLP5_logprobs, bottom_MLP5_ablated_logprobs, positive=True)
        top_neuron_pos_difference_logprobs, top_pos_indices = get_top_difference_neurons(all_MLP5_logprobs, top_MLP5_ablated_logprobs, positive=True)
        # Deboosted tokens
        bottom_neuron_neg_difference_logprobs, bottom_neg_indices = get_top_difference_neurons(all_MLP5_logprobs, bottom_MLP5_ablated_logprobs, positive=False)
        top_neuron_neg_difference_logprobs, top_neg_indices = get_top_difference_neurons(all_MLP5_logprobs, top_MLP5_ablated_logprobs, positive=False)

        bottom_pos_tokens = [model.to_str_tokens([i])[0] for i in bottom_pos_indices]
        top_pos_tokens = [model.to_str_tokens([i])[0] for i in top_pos_indices]
        bottom_neg_tokens = [model.to_str_tokens([i])[0] for i in bottom_neg_indices]
        top_neg_tokens = [model.to_str_tokens([i])[0] for i in top_neg_indices]

        summed_neuron_logprob_boosts[option][num_neurons]["Top"]["Boosted"] = {
            "Tokens": top_pos_tokens,
            "Logprob difference": top_neuron_pos_difference_logprobs.tolist()
        }
        summed_neuron_logprob_boosts[option][num_neurons]["Top"]["Deboosted"] = {
            "Tokens": top_neg_tokens,
            "Logprob difference": top_neuron_neg_difference_logprobs.tolist()
        }
        summed_neuron_logprob_boosts[option][num_neurons]["Bottom"]["Boosted"] = {
            "Tokens": bottom_pos_tokens,
            "Logprob difference": bottom_neuron_pos_difference_logprobs.tolist()
        }
        summed_neuron_logprob_boosts[option][num_neurons]["Bottom"]["Deboosted"] = {
            "Tokens": bottom_neg_tokens,
            "Logprob difference": bottom_neuron_neg_difference_logprobs.tolist()
        }

with open('data/verify_bigrams/summed_neuron_boosts.json', 'w') as f:
    json.dump(summed_neuron_logprob_boosts, f)
# %%
