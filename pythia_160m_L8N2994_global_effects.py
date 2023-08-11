#%%
import torch
from transformer_lens import HookedTransformer
from jaxtyping import Float, Int, Bool
from torch import Tensor
from tqdm.auto import tqdm
import plotly.io as pio
import haystack_utils
import scipy.stats as stats
import numpy as np
import pandas as pd
import plotly.express as px
from statsmodels.nonparametric.smoothers_lowess import lowess

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

def get_space_tokens(tokens = None):
    if tokens is None:
        tokens = [i for i in range(model.cfg.d_vocab)]
    space_tokens = []
    non_space_tokens = []
    ignore_tokens = all_ignore.flatten().tolist()
    for i in tokens:
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

def get_space_tokens_diff(space_tokens, non_space_tokens, original_probs, ablated_probs):
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

# %%
space_token_diffs = []
non_space_token_diffs = []
for prompt in tqdm(german_data):
    with model.hooks(snap_pos_to_peak_1_hook):
        peak_1_prob = model(prompt, return_type="logits").softmax(dim=-1)
    with model.hooks(snap_pos_to_peak_2_hook):
        peak_2_prob = model(prompt, return_type="logits").softmax(dim=-1)
    space_token_diff, non_space_token_diff = get_space_tokens_diff(space_tokens, non_space_tokens, peak_1_prob[0], peak_2_prob[0])
    space_token_diffs.append(space_token_diff)
    non_space_token_diffs.append(non_space_token_diff)
print(f"Space token diff: {np.mean(space_token_diffs):.8f}, non space token diff: {np.mean(non_space_token_diffs):.8f}")
# %%

full_german_data = haystack_utils.load_json_data("data/german_europarl.json")
top_counts, top_tokens = haystack_utils.get_common_tokens(full_german_data, model, all_ignore, k=model.cfg.d_vocab_out, return_counts=True)

# %%
sorted_counts = top_counts.tolist()
sorted_counts.sort(reverse=True)
px.line(sorted_counts)
# %%
common_german_tokens = top_tokens[:2000].tolist()
# %%
german_space_tokens, german_non_space_tokens = get_space_tokens(common_german_tokens)
print(len(german_space_tokens), len(german_non_space_tokens))
# %%

space_token_diffs = []
non_space_token_diffs = []
for prompt in german_data:
    with model.hooks(snap_pos_to_peak_1_hook):
        peak_1_prob = model(prompt, return_type="logits").softmax(dim=-1)
    with model.hooks(snap_pos_to_peak_2_hook):
        peak_2_prob = model(prompt, return_type="logits").softmax(dim=-1)
    space_token_diff, non_space_token_diff = get_space_tokens_diff(german_space_tokens, german_non_space_tokens, peak_1_prob[0], peak_2_prob[0])
    space_token_diffs.append(space_token_diff)
    non_space_token_diffs.append(non_space_token_diff)
print(f"Space token diff: {np.mean(space_token_diffs):.8f}, non space token diff: {np.mean(non_space_token_diffs):.8f}")

# %%

dla = model.W_out[LAYER, NEURON] @ model.unembed.W_U

px.histogram(dla.flatten().cpu().numpy())
# %%
space_dla = dla[german_space_tokens]
px.line(space_dla.flatten().cpu().numpy())
# %%
non_space_dla = dla[german_non_space_tokens]
px.line(non_space_dla.flatten().cpu().numpy())
# %%


# Create a Pandas Series from your data
data = pd.Series(non_space_dla.flatten().cpu().numpy())
# Apply a moving average with a window size of your choice (e.g., 5)
non_space_smoothed_data = data.rolling(window=5).mean()
data = pd.Series(space_dla.flatten().cpu().numpy())
# Apply a moving average with a window size of your choice (e.g., 5)
space_smoothed_data = data.rolling(window=5).mean()

df = pd.DataFrame({
    'Space_Smoothed': space_smoothed_data,
    'Non_Space_Smoothed': non_space_smoothed_data
})

# Plot the smoothed line
fig = px.line(df)
fig.show()
# %%
space_token_counts = top_counts[german_space_tokens].cpu().numpy()
non_space_token_counts = top_counts[german_non_space_tokens].cpu().numpy()

# %%
# Find the indices where top_tokens equals count_tokens
indices = torch.where(torch.isin(top_tokens.cpu(), german_space_tokens.cpu()))[0]
space_token_counts = top_counts[indices].cpu().numpy()
indices = torch.where(torch.isin(top_tokens.cpu(), german_non_space_tokens.cpu()))[0]
non_space_token_counts = top_counts[indices].cpu().numpy()
# %%
df = pd.DataFrame({
    'space_token_counts': pd.Series(space_token_counts),
    'non_space_token_counts': pd.Series(non_space_token_counts)
})

# Plot the smoothed line
fig = px.line(df)
fig.show()
# %%

# New investigation

token_counts = torch.zeros(model.cfg.d_vocab).cuda()
for example in tqdm(german_data):
    tokens = model.to_tokens(example)
    for token in tokens[0]:
        token_counts[token.item()] += 1
# %%
full_german_data = haystack_utils.load_json_data("data/german_europarl.json")
token_counts, top_tokens = haystack_utils.get_token_counts(full_german_data, model)
token_counts = token_counts.cpu()
# %%
sorted_counts = token_counts.tolist()[:3000]
sorted_counts.sort(reverse=True)
px.line(sorted_counts)

# %%
#common_german_tokens = torch.arange(start=0, end=model.cfg.d_vocab, step=1)[token_counts > 1].tolist()
tmp_counts, common_german_tokens = haystack_utils.get_common_tokens(german_data, model, all_ignore, k=model.cfg.d_vocab, return_counts=True)
common_german_tokens = common_german_tokens[tmp_counts > 0].tolist()
german_space_tokens, german_non_space_tokens = get_space_tokens(common_german_tokens)
print(len(german_space_tokens), len(german_non_space_tokens))
# %%
dla = model.W_out[LAYER, NEURON] @ model.unembed.W_U
space_dla = dla[german_space_tokens]
non_space_dla = dla[german_non_space_tokens]

# %%
data = []

for token in german_space_tokens.tolist():
    data.append([token, True, dla[token].item(), token_counts[token].item()])
for token in german_non_space_tokens.tolist():
    data.append([token, False, dla[token].item(), token_counts[token].item()])

df = pd.DataFrame(data, columns=["token", "is_space", "dla", "count"])
# %%
px.scatter(df, y="dla", x="count", color="is_space", log_x=True, title="DLA for space and non_space tokens")

# %%
df = pd.DataFrame(data, columns=["token", "is_space", "dla", "count"])
df = df.sort_values(by=["is_space", "count"], ascending=False)
grouped = df.groupby(["is_space", "count"]).mean().reset_index()
px.line(grouped, y="dla", x="count", color="is_space", log_x=True, title="DLA for space and non_space tokens")
# %%


df = pd.DataFrame(data, columns=["token", "is_space", "dla", "count"])
df = df.sort_values(by=["is_space", "count"], ascending=False)
grouped = df.groupby(["is_space", "count"]).mean().reset_index()

# Apply lowess smoothing
smoothed_lines = []
for is_space_val in grouped['is_space'].unique():
    subset = grouped[grouped['is_space'] == is_space_val]
    smooth_line = lowess(subset['dla'], subset['count'], frac=0.3)
    smoothed_lines.append(pd.DataFrame(smooth_line, columns=['count', 'dla']))
    smoothed_lines[-1]['is_space'] = is_space_val

smoothed_df = pd.concat(smoothed_lines)
fig = px.line(smoothed_df, y="dla", x="count", color="is_space", log_x=True, title="DLA for space and non_space tokens (log_x, smoothed)")
fig.show()

# %%
