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
import plotly.graph_objects as go
import numpy as np
import pandas as pd


pio.renderers.default = "notebook_connected"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)

from haystack_utils import get_mlp_activations
import haystack_utils

%reload_ext autoreload
%autoreload 2
# %%
ngram = "orschlägen"
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device)

# %%
activate_neurons_fwd_hooks, deactivate_neurons_fwd_hooks = haystack_utils.get_context_ablation_hooks(3, [669], model)
all_ignore, _ = haystack_utils.get_weird_tokens(model, plot_norms=False)
# %%
german_data = haystack_utils.load_json_data("data/german_europarl.json")[:200]
common_tokens = haystack_utils.get_common_tokens(german_data, model, all_ignore, k=100)
# %%
# Sort tokens into new word vs continuation
new_word_tokens = []
continuation_tokens = []
for token in common_tokens:
    str_token = model.to_single_str_token(token.item())
    if str_token.startswith(" "):
        new_word_tokens.append(token)
    else:
        continuation_tokens.append(token)
new_word_tokens = torch.stack(new_word_tokens)
continuation_tokens = torch.stack(continuation_tokens)
print(new_word_tokens.shape, continuation_tokens.shape)
print(model.to_str_tokens(new_word_tokens))
print(model.to_str_tokens(continuation_tokens))
# %%
prompts = haystack_utils.generate_random_prompts(ngram, model, common_tokens, 200, length=20)

# Create orsch-X prompts
test_prompts = haystack_utils.replace_column(prompts, -2, continuation_tokens)
print(model.to_str_tokens(test_prompts[0]))

_, test_cache = model.run_with_cache(test_prompts)
res_cache_prev = test_cache['blocks.4.hook_resid_post'][:, -2]

# Create Y-X prompts
random_prompts = haystack_utils.replace_column(test_prompts, -3, continuation_tokens)
print(model.to_str_tokens(random_prompts[0]))

_, random_cache = model.run_with_cache(random_prompts)
res_cache_random = random_cache['blocks.4.hook_resid_post'][:, -2]

# %%
prev_token_direction = (res_cache_prev - res_cache_random).mean(0)
print(prev_token_direction[:10])
# %%

# Create X-lä prompts
test_prompts = haystack_utils.replace_column(prompts, -3, continuation_tokens)
print(model.to_str_tokens(test_prompts[0]))

_, test_cache = model.run_with_cache(test_prompts)
res_cache_curr = test_cache['blocks.4.hook_resid_post'][:, -2]

# Create X-Y prompts
random_prompts = haystack_utils.replace_column(test_prompts, -2, continuation_tokens)
print(model.to_str_tokens(random_prompts[0]))

_, random_cache = model.run_with_cache(random_prompts)
res_cache_random = random_cache['blocks.4.hook_resid_post'][:, -2]

# %% 
curr_token_direction = (res_cache_curr - res_cache_random).mean(0)
print(curr_token_direction[:10])
# %%
context_direction = model.W_out[3, 669, :]

# %%
context_direction.shape, prev_token_direction.shape, curr_token_direction.shape
# %%
cosine = torch.nn.CosineSimilarity(dim=1)

mlp_5 = model.W_in[5].T

prev_token_sim = cosine(mlp_5, prev_token_direction.unsqueeze(0))
curr_token_sim = cosine(mlp_5, curr_token_direction.unsqueeze(0))
context_sim = cosine(mlp_5, context_direction.unsqueeze(0))
print(prev_token_sim.shape, curr_token_sim.shape, context_sim.shape)

# %%

def plot_histogram(t1, t2, t3, name1, name2, name3):
    t1 = t1.cpu().numpy()
    t2 = t2.cpu().numpy()
    t3 = t3.cpu().numpy()
    fig = go.Figure()
    bin_width= 0.01
    fig.add_trace(go.Histogram(x=t1, name=name1, opacity=0.5, histnorm='probability density', xbins=dict(size=bin_width)))
    fig.add_trace(go.Histogram(x=t2, name=name2, opacity=0.5 , histnorm='probability density', xbins=dict(size=bin_width)))
    fig.add_trace(go.Histogram(x=t3, name=name3, opacity=0.5, histnorm='probability density', xbins=dict(size=bin_width)))
    
    fig.update_layout(
        title="Individual MLP5 similarities to direction vectors",
        xaxis_title="Cosine Similarity",
        yaxis_title="Probability Density",
        barmode="overlay",
    )
    
    fig.show()

plot_histogram(prev_token_sim, curr_token_sim, context_sim, "Previous Token", "Current Token", "Context")
# %%

cutoff = 0.05
prev_token_neurons = torch.where(prev_token_sim > cutoff)[0]
curr_token_neurons = torch.where(curr_token_sim > cutoff)[0]
context_neuron_neurons = torch.where(context_sim > cutoff)[0]
print(prev_token_neurons.shape, curr_token_neurons.shape, context_neuron_neurons.shape)

# %%
prev_and_curr = np.intersect1d(prev_token_neurons.cpu().numpy(), curr_token_neurons.cpu().numpy())
all = np.intersect1d(prev_and_curr, context_neuron_neurons.cpu().numpy())
print(prev_and_curr.shape, all.shape)
print(all)
# %%

for neuron in all:
    print(prev_token_sim[neuron].item(), curr_token_sim[neuron].item(), context_sim[neuron].item())
# %%

# Cache neuron activations for different settings
# Context active / disabled
# Prev token active / disabled
# Current token active / disabled

data = {
    "Neuron": [], 
    "PrevTokenPresent": [], 
    "CurrTokenPresent": [],
    "ContextPresent": [],
    "Activation": [],
    }

prompts = haystack_utils.generate_random_prompts(ngram, model, common_tokens, 200, length=20)
cache_name = "blocks.5.mlp.hook_pre"
neurons = torch.LongTensor([i for i in range(2048)])#all
neuron_list = neurons.tolist()

def add_to_data(data: dict[str, list], cache_result: Tensor, neurons, prev_token: bool, curr_token: bool, context_neuron: bool):
    data["Neuron"].extend(neuron_list)
    data["PrevTokenPresent"].extend([prev_token] * len(neurons))
    data["CurrTokenPresent"].extend([curr_token] * len(neurons))
    data["ContextPresent"].extend([context_neuron] * len(neurons))
    data["Activation"].extend(cache_result[neurons].tolist())
    return data

# Run on orsch-lä prompts
print(model.to_str_tokens(prompts[0]))
_, original_cache = model.run_with_cache(prompts)
with model.hooks(fwd_hooks=deactivate_neurons_fwd_hooks):
    _, original_cache_ablated = model.run_with_cache(prompts)
res_cache_orig = original_cache[cache_name][:, -2].mean(0)
res_cache_orig_ablated = original_cache_ablated[cache_name][:, -2].mean(0)
data = add_to_data(data, res_cache_orig, neurons, prev_token=True, curr_token=True, context_neuron=True)
data = add_to_data(data, res_cache_orig_ablated, neurons, prev_token=True, curr_token=True, context_neuron=False)

# Create orsch-X prompts
test_prompts = haystack_utils.replace_column(prompts, -2, continuation_tokens)
print(model.to_str_tokens(test_prompts[0]))

_, test_cache = model.run_with_cache(test_prompts)
with model.hooks(fwd_hooks=deactivate_neurons_fwd_hooks):
    _, test_cache_ablated = model.run_with_cache(test_prompts)
res_cache_prev = test_cache[cache_name][:, -2].mean(0)
res_cache_prev_ablated = test_cache_ablated[cache_name][:, -2].mean(0)
data = add_to_data(data, res_cache_prev, neurons, prev_token=True, curr_token=False, context_neuron=True)
data = add_to_data(data, res_cache_prev_ablated, neurons, prev_token=True, curr_token=False, context_neuron=False)

# Create X-lä prompts
test_prompts = haystack_utils.replace_column(prompts, -3, continuation_tokens)
print(model.to_str_tokens(test_prompts[0]))

_, test_cache = model.run_with_cache(test_prompts)
with model.hooks(fwd_hooks=deactivate_neurons_fwd_hooks):
    _, test_cache_ablated = model.run_with_cache(test_prompts)
res_cache_curr = test_cache[cache_name][:, -2].mean(0)
res_cache_curr_ablated = test_cache_ablated[cache_name][:, -2].mean(0)
data = add_to_data(data, res_cache_curr, neurons, prev_token=False, curr_token=True, context_neuron=True)
data = add_to_data(data, res_cache_curr_ablated, neurons, prev_token=False, curr_token=True, context_neuron=False)

# Create Y-X prompts
random_prompts = haystack_utils.replace_column(test_prompts, -3, continuation_tokens)
random_prompts = haystack_utils.replace_column(random_prompts, -2, continuation_tokens)
print(model.to_str_tokens(random_prompts[0]))

_, random_cache = model.run_with_cache(random_prompts)
with model.hooks(fwd_hooks=deactivate_neurons_fwd_hooks):
    _, random_cache_ablated = model.run_with_cache(random_prompts)
res_cache_random = random_cache[cache_name][:, -2].mean(0)
res_cache_random_ablated = random_cache_ablated[cache_name][:, -2].mean(0)
data = add_to_data(data, res_cache_random, neurons, prev_token=False, curr_token=False, context_neuron=True)
data = add_to_data(data, res_cache_random_ablated, neurons, prev_token=False, curr_token=False, context_neuron=False)

df = pd.DataFrame(data)

# %%
print("Original activation:", np.round(res_cache_orig[neurons].tolist(), 2))
print("Prev token activation:", np.round(res_cache_prev[neurons].tolist(), 2))
print("Curr token activation:", np.round(res_cache_curr[neurons].tolist(), 2))
print("Random activation:", np.round(res_cache_random[neurons].tolist(), 2))

print("Original ablated activation:", np.round(res_cache_orig_ablated[neurons].tolist(), 2))
print("Prev token ablated activation:", np.round(res_cache_prev_ablated[neurons].tolist(), 2))
print("Curr token ablated activation:", np.round(res_cache_curr_ablated[neurons].tolist(), 2))
print("Random ablated activation:", np.round(res_cache_random_ablated[neurons].tolist(), 2))

# %%

# Save huge jsons
# MLP5 caches
# Neuron wise cosine sims for prev / current / context
# Create heatmap vis
len(df)
print(df.head())
# %%
# Convert boolean values to string and combine them
df['Prev/Curr/Context'] = df.apply(lambda row: ("Y" if row['PrevTokenPresent'] else "N")+("Y"if row['CurrTokenPresent'] else "N")+("Y" if row['ContextPresent'] else "N"), axis=1)

# Pivot the dataframe
pivot_df = df.pivot(index='Neuron', columns='Prev/Curr/Context', values='Activation')
pivot_df = pivot_df.round(2)
pivot_df["AND"] = (pivot_df["YYY"]>0) & (pivot_df["YYN"]<0) & (pivot_df["YNY"]<0) & (pivot_df["NYY"]<0)
pivot_df = pivot_df.sort_values(by="AND", ascending=False)
pivot_df[:20]

# %%

neurons = torch.LongTensor(pivot_df[pivot_df["AND"]].index.tolist()).cuda()
mean_activations = torch.Tensor(pivot_df[pivot_df["AND"]]["NNN"].tolist()).cuda()
# %%

def ablate_mlp_5_hook(value, hook):
    value[:, :, neurons] = mean_activations
    return value

loss = model(prompts, return_type="loss", loss_per_token=True)[:, -1].mean().item()

with model.hooks(fwd_hooks=[("blocks.5.mlp.hook_pre", ablate_mlp_5_hook)]):
    ablated_loss = model(prompts, return_type="loss", loss_per_token=True)[:, -1].mean().item()

print(loss, ablated_loss)
# %%
