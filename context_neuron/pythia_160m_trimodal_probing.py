# %%
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import plotly.io as pio
import pandas as pd
import numpy as np
import plotly.express as px 
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import haystack_utils
import hook_utils
import probing_utils
from probing_utils import get_and_score_new_word_probe

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

LAYER, NEURON = 8, 2994

hook_name = f'blocks.{LAYER}.mlp.hook_post'
# %%
# Trimodal neuron only
print(get_and_score_new_word_probe(model, german_data, hook_name))
# %%
print(get_and_score_new_word_probe(model, german_data, hook_name, np.s_[0, :-1, :]))
# %%
activation_slice = np.s_[0, :-1, [neuron for neuron in range(model.cfg.d_mlp) if neuron != NEURON]]
print(get_and_score_new_word_probe(model, german_data, hook_name, activation_slice)) # .888 f1
# %%
activation_slice = np.s_[0, :-1, :]
x, y = probing_utils.get_new_word_labels_and_activations(model, german_data, hook_name, activation_slice)
# %%
one_sparse_probe_scores = []
for i in tqdm(range(model.cfg.d_mlp)):
    probe = probing_utils.get_probe(x[:20_000, [i]], y[:20_000])
    f1, mcc = probing_utils.get_probe_score(probe, x[20_000:, [i]], y[20_000:])
    one_sparse_probe_scores.append([f1, mcc])

one_sparse_probe_scores_df = pd.DataFrame(one_sparse_probe_scores, columns=["f1", "mcc"])
with open(f'data/pythia_160m/layer_8/single_neurons_df.pkl', 'wb') as f:
    pickle.dump(one_sparse_probe_scores_df, f)

# %%
with open(f'data/pythia_160m/layer_8/single_neurons_df.pkl', 'rb') as f:
    one_sparse_probe_scores_df = pickle.load(f)

top_one_neurons = one_sparse_probe_scores_df.sort_values(by="mcc", ascending=False).head(10).index.tolist()
two_sparse_probe_scores = []
for i in top_one_neurons:
    for j in tqdm(range(model.cfg.d_mlp)):
        if i == j:
            continue
        probe = probing_utils.get_probe(x[:20_000, [i, j]], y[:20_000])
        f1, mcc = probing_utils.get_probe_score(probe, x[20_000:, [i, j]], y[20_000:])
        two_sparse_probe_scores.append([f1, mcc, i, j])

two_sparse_probe_scores_df = pd.DataFrame(two_sparse_probe_scores, columns=["f1", "mcc", "neuron_1", "neuron_2"])
with open(f'data/pythia_160m/layer_8/probes_two_sparse_df_10_mcc.pkl', 'wb') as f:
    pickle.dump(two_sparse_probe_scores_df, f)

# %%
with open(f'data/pythia_160m/layer_8/single_neurons_df.pkl', 'rb') as f:
    one_sparse_probe_scores_df = pickle.load(f)
top_one_neurons_by_f1 = one_sparse_probe_scores_df.sort_values(by="f1", ascending=False).head(10).index.tolist()
two_sparse_probe_scores = []
for i in top_one_neurons_by_f1:
    for j in tqdm(range(model.cfg.d_mlp)):
        if i == j:
            continue
        probe = probing_utils.get_probe(x[:20_000, [i, j]], y[:20_000])
        f1, mcc = probing_utils.get_probe_score(probe, x[20_000:, [i, j]], y[20_000:])
        two_sparse_probe_scores.append([f1, mcc, i, j])

two_sparse_probe_scores_df = pd.DataFrame(two_sparse_probe_scores, columns=["f1", "mcc", "neuron_1", "neuron_2"])
with open(f'data/pythia_160m/layer_8/probes_two_sparse_df_10_f1.pkl', 'wb') as f:
    pickle.dump(two_sparse_probe_scores_df, f)
# %%
scores = []
activation_slice = np.s_[0, :-1, :]
for layer in range(12):
    hook_name = f'blocks.{layer}.hook_mlp_out'
    scores.append(get_and_score_new_word_probe(model, german_data, hook_name, activation_slice))

df = pd.DataFrame(scores, columns=["f1", "mcc"])
fig = haystack_utils.line(df, title="F1 and MCC Scores at MLP out by layer", xlabel="Layer", ylabel="F1 Score")
# %%
cosine_sim = torch.nn.CosineSimilarity(dim=0)
# mean ablate context neuron, train at hook out, compare cosine sim with context neuron
neuron_out = model.W_out[8, 2994] # 768
activation_slice = np.s_[0, :-1, :]

mlp_out_sims = []
with model.hooks([hook_utils.get_ablate_neuron_hook(8, 2994, 4.5)]):
    for layer in range(12):
        hook_name = f'blocks.{layer}.hook_mlp_out'
        x, y = probing_utils.get_new_word_labels_and_activations(model, german_data, hook_name, activation_slice)
        probe = probing_utils.get_probe(x[:20_000], y[:20_000])
        mlp_out_sims.append(cosine_sim(torch.from_numpy(probe.coef_[0]).cuda(), neuron_out).cpu())

# %%
haystack_utils.line([sim.cpu() for sim in mlp_out_sims], title='Cosine sim between MLP out probe at each layer and context neuron')
# %%
# Dense probe performance on each residual and MLP out
# We can do each MLP out using the current method
# Need a new method for the dimensions of the residual
# %%
from probing_utils import get_new_word_labels_and_resid_activations
activations_dict, labels_dict = get_new_word_labels_and_resid_activations(model, german_data)
scores = []
resid_sims = []
with model.hooks([hook_utils.get_ablate_neuron_hook(8, 2994, 4.5)]):
    for i in range(len(activations_dict.items())):
        component = activations_dict[i]
        labels = labels_dict[i]
        probe = probing_utils.get_probe(component[:20_000], labels[:20_000])
        resid_sims.append(cosine_sim(torch.from_numpy(probe.coef_[0]).cuda(), neuron_out).cpu())
        scores.append(probing_utils.get_probe_score(probe, component[20_000:], labels[20_000:]))

_, cache = model.run_with_cache(german_data[0])
_, component_labels = cache.decompose_resid(apply_ln=False, return_labels=True)
haystack_utils.line(scores, xticks=component_labels, title="F1 Score at residual stream by layer", xlabel="Layer", ylabel="F1 Score")
haystack_utils.line(resid_sims, xticks=component_labels, title='Cosine sim between resid probe at each layer and context neuron')

# %%
# Dense probe on whether the residual stream is in German or not, try ablating that direction
x, y = probing_utils.get_is_german_labels_and_resid_activations(model, german_data, english_data, 8)
probe = probing_utils.get_probe(component[:20_000], labels[:20_000])
probe_coefs = torch.from_numpy(probe.coef_[0]).cuda().float()

hook_name = f'blocks.8.hook_resid_pre'
def hook_fn(value, hook):
    value = value - torch.einsum('k,ijk->ijk', probe_coefs, value)
    return value
hooks = [(hook_name, hook_fn)]
def is_german_ablation():
    with model.hooks(hooks + [('blocks.8.mlp.hook_post', hook_utils.save_activation)]):
        original_loss = haystack_utils.get_average_loss(german_data, model)
    with model.hooks(hooks + [hook_utils.get_snap_to_peak_1_hook()]):
        ablated_loss = haystack_utils.get_average_loss(german_data, model)
    with model.hooks(hooks + [hook_utils.get_snap_to_peak_2_hook()]):
        second_ablated_loss = haystack_utils.get_average_loss(german_data, model)
    print(original_loss, ablated_loss, second_ablated_loss)

is_german_ablation()

acts = model.hook_dict['blocks.8.mlp.hook_post'].ctx['activation']
print(acts)
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
LAYER, NEURON = 8, 2994
cosine_sim = torch.nn.CosineSimilarity(dim=0)
ctx_neuron_sims = [
    cosine_sim(model.W_out[LAYER, NEURON, :], model.W_in[LAYER, :, NEURON]).item(),
    cosine_sim(model.W_out[5, 2649, :], model.W_in[5, :, 2649]).item(),
    cosine_sim(model.W_in[5, :, 2649], model.W_in[LAYER, :, NEURON]).item(),
    cosine_sim(model.W_out[5, 2649, :], model.W_out[LAYER, NEURON, :]).item(),
    cosine_sim(model.W_out[5, 2649, :], model.W_in[LAYER, :, NEURON]).item(),
    cosine_sim(model.W_in[5, :, 2649], model.W_out[LAYER, NEURON, :]).item(),
]
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
# import plotly.graph_objects as go
# import networkx as nx

# # Create a directed graph using NetworkX
# G = nx.DiGraph()
# G.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 4)])
# weights = [0.5, 0.2, 0.9, 0.7]

# # Create positions for nodes
# pos = nx.spring_layout(G)

# # Extract coordinates
# x_nodes = [pos[k][0] for k in pos]
# y_nodes = [pos[k][1] for k in pos]

# # Create edge traces
# edge_x = []
# edge_y = []
# edge_width = []
# for edge, weight in zip(G.edges(), weights):
#     x0, y0 = pos[edge[0]]
#     x1, y1 = pos[edge[1]]
#     edge_x += [x0, x1, None]
#     edge_y += [y0, y1, None]
#     edge_width.append(weight)

# # Plot nodes and edges
# node_trace = go.Scatter(x=x_nodes, y=y_nodes, mode='markers')
# edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=edge_width))
# fig = go.Figure(data=[edge_trace, node_trace])

# # Show plot
# fig.show()
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
activation_slice = np.s_[0, :-1, [1, 2]] # Random baseline
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

# print(german_data[1])
# model.generate(german_data[1], max_new_tokens=100)

# %%
# Context neuron activations with L5 context neuron ablated

# Check out the learned weights
# hook_name = f'blocks.{LAYER}.mlp.hook_post'
# activation_slice = np.s_[0, :-1, [neuron for neuron in range(model.cfg.d_mlp) if neuron != NEURON]]
# x, y = get_new_word_labels_and_activations(model, german_data, hook_name, activation_slice)
# probe = get_new_word_dense_probe(x, y)
# haystack_utils.line(np.sort(np.absolute(probe.coef_[0])), title="Probe Coefficients", xlabel="Neuron", ylabel="Coefficient")