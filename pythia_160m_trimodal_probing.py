# %%
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import plotly.io as pio
import pandas as pd
import numpy as np
import plotly.express as px 
from collections import defaultdict
from functools import partial

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import haystack_utils
import hook_utils
import plotting_utils
import probing_utils
from probing_utils import get_and_score_new_word_probe
from sklearn import preprocessing

pio.renderers.default = "notebook_connected+notebook"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)

# %%
%reload_ext autoreload
%autoreload 2

model = HookedTransformer.from_pretrained("EleutherAI/pythia-160m",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device)

german_data = haystack_utils.load_json_data("data/german_europarl.json")[:200]
english_data = haystack_utils.load_json_data("data/english_europarl.json")[:200]

LAYER, NEURON = 8, 2994

# %%
# Check out the learned weights
# hook_name = f'blocks.{LAYER}.mlp.hook_post'
# activation_slice = np.s_[0, :-1, [neuron for neuron in range(model.cfg.d_mlp) if neuron != NEURON]]
# x, y = get_new_word_labels_and_activations(model, german_data, hook_name, activation_slice)
# probe = get_new_word_dense_probe(x, y)
# haystack_utils.line(np.sort(np.absolute(probe.coef_[0])), title="Probe Coefficients", xlabel="Neuron", ylabel="Coefficient")
# %%
# Trimodal neuron only
hook_name = f'blocks.{LAYER}.mlp.hook_post'
# x, y = get_new_word_labels_and_activations(model, german_data, hook_name)
# f1, mcc = get_new_word_dense_probe_score(x, y)
# print(f1, mcc) # f1 0.799
print(get_and_score_new_word_probe(model, german_data, hook_name))
# %%
# The final position's activation has no label and is excluded
activation_slice = np.s_[0, :-1, [neuron for neuron in range(model.cfg.d_mlp) if neuron != NEURON]]
print(get_and_score_new_word_probe(model, german_data, hook_name, activation_slice)) # .888 f1
# %%
for i in range(model.cfg.d_mlp):
    activation_slice = np.s_[0, :-1, [i]]
    print(get_and_score_new_word_probe(model, german_data, hook_name, activation_slice)) # .888 f1, 0.94 mcc
# %% 
scores = []
activation_slice = np.s_[0, :-1, :]
for layer in range(11):
    hook_name = f'blocks.{layer}.hook_mlp_out'
    scores.append(get_and_score_new_word_probe(model, german_data, hook_name, activation_slice))

df = pd.DataFrame(scores, columns=["f1", "mcc"])
fig = haystack_utils.line(df, title="F1 and MCC Scores at MLP out by layer", xlabel="Layer", ylabel="F1 Score")


# %%
# Dense probe performance on each residual and MLP out
# We can do each MLP out using the current method
# Need a new method for the dimensions of the residual
# %%
from probing_utils import get_new_word_labels_and_resid_activations
activations_dict, labels_dict = get_new_word_labels_and_resid_activations(model, german_data)

scores = []
for i in range(len(activations_dict.items())):
    component = activations_dict[i]
    labels = labels_dict[i]

    probe = probing_utils.get_probe(component[:20_000], labels[:20_000])
    scores.append(probing_utils.get_probe_score(probe, component[20_000:], labels[20_000:]))

# %% 

_, cache = model.run_with_cache(german_data[0])
_, component_labels = cache.decompose_resid(apply_ln=False, return_labels=True)
haystack_utils.line(scores, xticks=component_labels, title="F1 Score at residual stream by layer", xlabel="Layer", ylabel="F1 Score")
# %%
l8_n2994_input = model.W_in[LAYER, :, NEURON].cpu().numpy()

projections = []
for component in activations_dict.values():
    projection = np.dot(component, l8_n2994_input)[:, np.newaxis]
    probe = probing_utils.get_probe(projection[:20_000], labels[:20_000])
    projections.append(probing_utils.get_probe_score(probe, projection[20_000:], labels[20_000:]))

print(projections)
haystack_utils.line(projections, xticks=component_labels, title="F1 Score of L8N2994 input direction in residual stream by layer", xlabel="Layer", ylabel="F1 Score")

# %%

# Other good context neuron == L5N2649
cosine_sim = torch.nn.CosineSimilarity(dim=0)
ctx_neuron_sims = []
ctx_neuron_sims.append(cosine_sim(model.W_out[LAYER, NEURON, :], model.W_in[LAYER, :, NEURON]).item())
ctx_neuron_sims.append(cosine_sim(model.W_out[5, 2649, :], model.W_in[5, :, 2649]).item())

ctx_neuron_sims.append(cosine_sim(model.W_in[5, :, 2649], model.W_in[LAYER, :, NEURON]).item())
ctx_neuron_sims.append(cosine_sim(model.W_out[5, 2649, :], model.W_out[LAYER, NEURON, :]).item())
ctx_neuron_sims.append(cosine_sim(model.W_out[5, 2649, :], model.W_in[LAYER, :, NEURON]).item())
ctx_neuron_sims.append(cosine_sim(model.W_in[5, :, 2649], model.W_out[LAYER, NEURON, :]).item())

sims = []
for neuron_i in range(model.cfg.d_mlp):
    sims.append(cosine_sim(model.W_out[5, 2649, :], model.W_in[LAYER, :, neuron_i]).item())
    sims.append(cosine_sim(model.W_out[5, 2649, :], model.W_out[LAYER, neuron_i, :]).item())
    sims.append(cosine_sim(model.W_in[5, :, 2649], model.W_in[LAYER, :, neuron_i]).item())
    sims.append(cosine_sim(model.W_in[5, :, 2649], model.W_out[LAYER, neuron_i, :]).item())
fig = px.histogram(sims, title="Cosine similarity between W_in/W_out of L5N2649 and L8 neurons. \
                   <br> Red lines are the similarity of L5N2649's W_in and W_out to L8N2994's W_in and W_out")

for sim in ctx_neuron_sims:
    fig.add_vline(x=sim, line_dash="dash", line_color="red")
fig.show()
# %%
# All layers
sims = []
for neuron_i in range(model.cfg.d_mlp):
    for layer_i in range(model.cfg.n_layers):
        sims.append(cosine_sim(model.W_out[5, 2649, :], model.W_in[layer_i, :, neuron_i]).item())
        sims.append(cosine_sim(model.W_out[5, 2649, :], model.W_out[layer_i, neuron_i, :]).item())
        sims.append(cosine_sim(model.W_in[5, :, 2649], model.W_in[layer_i, :, neuron_i]).item())
        sims.append(cosine_sim(model.W_in[5, :, 2649], model.W_out[layer_i, neuron_i, :]).item())
fig = px.histogram(sims, title="Cosine similarity between W_in/W_out of L5N2649 and L8 neurons. \
                   <br> Red lines are the similarity of L5N2649's W_in and W_out to L8N2994's W_in and W_out")

for sim in ctx_neuron_sims:
    fig.add_vline(x=sim, line_dash="dash", line_color="red")
fig.show()
# %%
# All high similarity neurons are context neurons except for L5N1162
for neuron_i in range(model.cfg.d_mlp):
    for layer_i in range(model.cfg.n_layers):
        for sim_i in range(4):
            current_sim = sims[neuron_i * (model.cfg.n_layers * 4) + layer_i * 4 + sim_i]
            if current_sim > 0.3:
                print(f'L{layer_i}N{neuron_i} Similarity {sim_i+1}/4', current_sim)

fig = px.histogram(sims, title="Cosine similarity between W_in/W_out of L5N2649 and L8 neurons. \
                   <br> Red lines are the similarity of L5N2649's W_in and W_out to L8N2994's W_in and W_out")

for sim in ctx_neuron_sims:
    fig.add_vline(x=sim, line_dash="dash", line_color="red")
fig.show()
# %%
hook_name = f'blocks.{5}.mlp.hook_post'
activation_slice = np.s_[0, :-1, [2649]]
print(get_and_score_new_word_probe(model, german_data, hook_name, activation_slice))

activation_slice = np.s_[0, :-1, [1162]]
print(get_and_score_new_word_probe(model, german_data, hook_name, activation_slice))

activation_slice = np.s_[0, :-1, [2649, 1162]]
print(get_and_score_new_word_probe(model, german_data, hook_name, activation_slice))

# %%
activation_slice = np.s_[0, :-1, [1, 2]]
print(get_and_score_new_word_probe(model, german_data, hook_name, activation_slice))

# %%
# Won't work due to exp(model.cfg.d_mlp) directions and each direction
# takes a while. I feel like a hill climbing method should work starting
# from the basis directions and angling towards multiple with higher 
# metric but idk how to formulate?
# Seems basically related to basis dimensions with polytopes
def grid_search():
    lr_model = LogisticRegression()
    lr_model.fit(np.zeros((2, model.cfg.d_mlp)), np.array([0, 1]))
    
    activation_slice = np.s_[0, :-1, :]
    x, y = probing_utils.get_new_word_labels_and_activations(model, german_data, hook_name, activation_slice)
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)

    # Define bias range for grid search
    bias_range = np.linspace(-5, 5, 100)
    def get_score_with_best_bias(direction, x, y) -> LogisticRegression:
        lr_model.coef_ = direction
        best_score = -np.inf
        for bias in bias_range:
            lr_model.intercept_ = np.array([bias])
            preds = lr_model.predict(x)
            score = f1_score(y, preds)
            best_score = max(best_score, score)

        return best_score.item()

    dimension = model.cfg.d_mlp
    num_samples = 10_000
    # Sample directions uniformly from the unit sphere
    directions = torch.randn(num_samples, dimension)
    directions /= directions.norm(dim=1, keepdim=True)

    metric_values = []
    for i in tqdm(range(num_samples)):
        metric_values.append(get_score_with_best_bias(
            directions[i].unsqueeze(0).numpy(), x[:100], y[:100]))
    threshold = 0.8
    percentage_above_threshold = (torch.tensor(metric_values) > threshold).float().mean().item() * 100

    print(f"The percentage of directions with a metric value above {threshold} is {percentage_above_threshold}%")
    return metric_values

# metric_values = grid_search()
# %%
def get_space_tokens():
    space_tokens = []
    non_space_tokens = []
    for i in range(model.cfg.d_vocab):
        string_token = model.to_single_str_token(i)
        if not string_token:
            continue
        if string_token[0] in [" ", ",", ".", ":", ";", "!", "?"]:
            space_tokens.append(i)
        else:
            non_space_tokens.append(i)
    return torch.tensor(space_tokens), torch.tensor(non_space_tokens)

# Get "is space" direction in embeddings
space_tokens, non_space_tokens = get_space_tokens()
# %%
# The is_space direction is dissimilar to the L8N2994 context neuron direction (-0.07)
space_tokens_mean_unembed = model.W_U[:, space_tokens].mean(dim=1)
non_space_tokens_mean_unembed = model.W_U[:, non_space_tokens].mean(dim=1)
is_space_direction = space_tokens_mean_unembed - non_space_tokens_mean_unembed
print(cosine_sim(is_space_direction, model.W_out[LAYER, NEURON, :]).item())
# %%
# What about the learned probe?
hook_name = f'blocks.11.hook_resid_post'
activation_slice = np.s_[0, :-1, :]
x, y = probing_utils.get_new_word_labels_and_activations(model, german_data, hook_name, activation_slice)
l8_mlp_out_probe = probing_utils.get_new_word_dense_probe(x, y)
print(cosine_sim(is_space_direction, torch.tensor(l8_mlp_out_probe.coef_[0]).cuda()).item())
# %%
# The optimal L8 next token direction is positive but even less similar to the 
# context neuron direction (0.03)
hook_name = f'blocks.8.hook_mlp_out'
activation_slice = np.s_[0, :-1, :]
x, y = probing_utils.get_new_word_labels_and_activations(model, german_data, hook_name, activation_slice)
l8_mlp_out_probe = probing_utils.get_new_word_dense_probe(x, y)
print(cosine_sim(is_space_direction, torch.tensor(l8_mlp_out_probe.coef_[0]).cuda()).item())

# %%
# So what does the direction directly boost?
all_ignore, _ = haystack_utils.get_weird_tokens(model, plot_norms=False)
# %%
coefs = torch.tensor(l8_mlp_out_probe.coef_[0]).cuda()
def remove_direction(value, hook):
    value = value - value * coefs
    return value

remove_direction_hooks = [('blocks.8.hook_mlp_out', remove_direction)]
haystack_utils.get_boosted_tokens(german_data[:20], model, remove_direction_hooks, all_ignore, deboost=True)

# %%
sims = cosine_sim(model.W_out[LAYER, NEURON, :].unsqueeze(1), model.W_U.reshape(model.cfg.d_model, model.cfg.d_vocab))

values, indices = torch.topk(sims, 70)
print(values)
print(model.to_str_tokens(indices))
# %%
px.histogram(sims.cpu().numpy().flatten(), title="Cosine similarity between L8N2994 and all tokens")
# %%
downstream_components = [(f"blocks.{layer}.hook_{component}_out") for layer in [9, 10, 11] for component in ['mlp', 'attn']]

original_losses = []
ablated_losses = []
direct_effect = []
for prompt in german_data:
    tokens = model.to_tokens(prompt)
    original_loss, ablated_loss, direct_effect, indirect_effect_through_component = haystack_utils.get_direct_effect( \
        prompt, 
        model, 
        context_ablation_hooks=[hook_utils.get_snap_to_peak_1_hook()], 
        context_activation_hooks=[], 
        deactivated_components=downstream_components,
        activated_components=[],
        pos=-1, return_type="loss")
    original_losses.append(original_loss)
    ablated_losses.append(ablated_loss)
    direct_effect.append(direct_effect)
# %%

# %%
# Can get top 10,000 german tokens, take the elbow, then filter our positive class 
# for tokens starting with a space that are common in German. And create one more
# direction for is German in general
# tokens, counts = haystack_utils.get_common_tokens(german_data, model, k=10_000)
# %%
# def space_tokens_diff(space_tokens, non_space_tokens, original_logits, ablated_logits):
#     original_logit_diffs = original_logits[:, space_tokens] - original_logits[:, non_space_tokens]
#     ablated_logits_diffs = ablated_logits[:, space_tokens] - ablated_logits[:, non_space_tokens]
#     return ablated_logits_diffs - original_logit_diffs

# deactivate_context_hooks = [hook_utils.get_neuron_hook(8, 2994, 0)]
# original_logits = model(german_data[:50], return_type='logits')
# with model.hooks(deactivate_context_hooks):
#     ablated_logits = model(german_data[:50], return_type='logits')

# %%

print(german_data[1])
model.generate(german_data[1], max_new_tokens=100)

# %%
# Context neuron activations with L5 context neuron ablated

