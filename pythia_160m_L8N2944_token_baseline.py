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
import numpy as np

pio.renderers.default = "notebook_connected+notebook"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)



%reload_ext autoreload
%autoreload 2

# %%
model = HookedTransformer.from_pretrained("EleutherAI/pythia-160m",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device)

german_data = haystack_utils.load_json_data("data/german_europarl.json")
all_ignore, _ = haystack_utils.get_weird_tokens(model, plot_norms=False)
#common_tokens = haystack_utils.get_common_tokens(german_data, model, all_ignore, k=100)

LAYER, NEURON = 8, 2994
#neuron_activations = haystack_utils.get_mlp_activations(german_data, LAYER, model, neurons=torch.LongTensor([NEURON]), mean=False).flatten()

# %%
# Unigram statistics
all_counts = torch.zeros(model.cfg.d_vocab)
next_is_space_counts = torch.zeros(model.cfg.d_vocab)

for prompt in tqdm(german_data):
    tokens = model.to_tokens(prompt, prepend_bos=False).flatten().cpu()
    # 2. Align next and current tokens
    next_tokens = tokens[1:].clone()
    tokens = tokens[:-1]
    # 3. Apply is_space mask
    is_space = haystack_utils.get_next_token_punctuation_mask(next_tokens, model)    
    # 4. Store next_is_space per token
    all_counts.index_add_(0, tokens, torch.ones_like(tokens, dtype=torch.float))
    next_is_space_counts.index_add_(0, tokens, is_space.to(torch.float))

print(all_counts.sum(), next_is_space_counts.sum(), all_counts.sum() - next_is_space_counts.sum())
# %%
common_tokens = torch.argwhere(all_counts > 30).flatten().tolist()
print(len(common_tokens))
common_tokens_space_prob = next_is_space_counts[common_tokens] / all_counts[common_tokens]
common_tokens_space_prob = torch.sort(common_tokens_space_prob, descending=True)[0]
fig = px.line(common_tokens_space_prob.cpu().numpy(), title="Next_is_space probability for common German tokens")
fig.update_layout(yaxis_title="P(next_is_space)", xaxis_title="Token rank", showlegend=False)
# %%

# Unigram predictions are
# For each token, predict the more likely class
# Count correct and incorrect number of predictions
# Calculate F1
common_tokens = torch.argwhere(all_counts > 1).flatten().tolist()

next_is_space = next_is_space_counts[common_tokens]
next_is_not_space = all_counts[common_tokens] - next_is_space_counts[common_tokens]
predict_space = (next_is_space_counts[common_tokens] / all_counts[common_tokens]) > 0.5

TP = next_is_space[predict_space].sum().item()
FP = next_is_not_space[predict_space].sum().item()
TN = next_is_not_space[~predict_space].sum().item()
FN = next_is_space[~predict_space].sum().item()

f1_score = (2 * TP) / (2 * TP + FP + FN)
mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
print(f"Unigram F1 score: {f1_score:.4f}")
print(f"Unigram MCC: {mcc:.4f}")
print(f"Unigram accuracy: {(TP + TN) / (TP + TN + FP + FN):.4f}")
# %%
# Bigram statistics
d_vocab = model.cfg.d_vocab
all_counts = torch.zeros(d_vocab, d_vocab)
next_is_space_counts = torch.zeros(d_vocab, d_vocab)

for prompt in tqdm(german_data):
    tokens = model.to_tokens(prompt, prepend_bos=False).flatten().cpu()
    # 2. Align previous, current, and next tokens
    prev_tokens = tokens[:-2]
    current_tokens = tokens[1:-1]
    next_tokens = tokens[2:]
    # 3. Apply is_space mask
    is_space = haystack_utils.get_next_token_punctuation_mask(next_tokens, model).to(torch.float)
    # 4. Store next_is_space per trigram
    for pt, ct, space in zip(prev_tokens, current_tokens, is_space):
        all_counts[pt, ct] += 1
        if space:
            next_is_space_counts[pt, ct] += 1

# %%
common_tokens = torch.argwhere(all_counts > 1)
print(f"Number of bigrams: {common_tokens.shape[0]}")

next_is_space = next_is_space_counts[common_tokens]
next_is_not_space = all_counts[common_tokens] - next_is_space_counts[common_tokens]
predict_space = (next_is_space_counts[common_tokens] / all_counts[common_tokens]) > 0.5

TP = next_is_space[predict_space].sum().item()
FP = next_is_not_space[predict_space].sum().item()
TN = next_is_not_space[~predict_space].sum().item()
FN = next_is_space[~predict_space].sum().item()

f1_score = (2 * TP) / (2 * TP + FP + FN)
mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
print(f"Bigram F1 score: {f1_score:.4f}")
print(f"Bigram MCC: {mcc:.4f}")
print(f"Bigram accuracy: {(TP + TN) / (TP + TN + FP + FN):.4f}")

# %%

all_counts = torch.Tensor([100, 100, 100, 100, 100])
next_is_space_counts = torch.Tensor([100, 90, 50, 75, 25])
common_tokens = torch.LongTensor([0, 2, 3, 4])

next_is_space = next_is_space_counts[common_tokens]
next_is_not_space = all_counts[common_tokens] - next_is_space_counts[common_tokens]
predict_space = (next_is_space_counts[common_tokens] / all_counts[common_tokens]) >= 0.5
print(next_is_space, next_is_not_space, predict_space)

TP = next_is_space[predict_space].sum().item()
FP = next_is_not_space[predict_space].sum().item()
TN = next_is_not_space[~predict_space].sum().item()
FN = next_is_space[~predict_space].sum().item()

f1_score = (2 * TP) / (2 * TP + FP + FN)
mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
print(f"Unigram F1 score: {f1_score:.4f}")
print(f"Unigram MCC: {mcc:.4f}")
print(f"Unigram accuracy: {(TP + TN) / (TP + TN + FP + FN):.4f}")
# %%

# compute model activations for unigram statistic tokens
common_german_tokens = haystack_utils.get_common_tokens(german_data[:500], model, all_ignore, 500)

# %%
def get_token_activations(end_token: str, model):
    prompts = haystack_utils.generate_random_prompts(end_token, model, common_german_tokens, 400, length=20)
    _, cache = model.run_with_cache(prompts)
    #fig = px.histogram(cache[f"blocks.{LAYER}.mlp.hook_post"][:, -1, NEURON].cpu().numpy(), nbins=50)
    #fig.show()
    return cache[f"blocks.{LAYER}.mlp.hook_post"][:, -1, NEURON].mean().item()


get_token_activations(" der", model)
# %%

# %%

data = []
common_tokens = torch.argwhere(all_counts > 100).flatten().tolist()
print(len(common_tokens))
for i, token in tqdm(enumerate(common_tokens)):
    #print(token, model.to_single_str_token(token))
    next_is_space_prob = common_tokens_space_prob[i].item()
    average_activation = get_token_activations(model.to_single_str_token(token), model)
    count = all_counts[token].item()
    data.append([token, next_is_space_prob, average_activation, count])

df = pd.DataFrame(data, columns=["token", "next_is_space_prob", "average_activation", "count"])
# %%
px.histogram(df, x="next_is_space_prob", nbins=100)
#%% 
px.histogram(df, x="average_activation", nbins=100)
# %%
px.scatter(df, y="next_is_space_prob", x="average_activation")

# %%
# Plot german text activations
activations = []
for prompt in tqdm(german_data[:100]):
    _, cache = model.run_with_cache(prompt)
    activations.extend(cache[f"blocks.{LAYER}.mlp.hook_post"][:, :, NEURON].flatten().tolist())
# add probability
fig = px.histogram(activations, nbins=200, title="L8N2994 on German text", histnorm="probability")
fig.update_layout(xaxis_title="Activation", yaxis_title="Probability")
fig.show()
# %%
common_german_tokens = haystack_utils.get_common_tokens(german_data[:500], model, all_ignore, 500)
#%%
activations = []
for i in range(50):
    prompt = haystack_utils.generate_random_prompts(" und", model, common_german_tokens, 50, length=100)[:, :-1]
    _, cache = model.run_with_cache(prompt)
    activations.extend(cache[f"blocks.{LAYER}.mlp.hook_post"][:, :, NEURON].flatten().tolist())
fig = px.histogram(activations, nbins=200, title="L8N2994 on random German tokens", histnorm="probability")
fig.update_layout(xaxis_title="Activation", yaxis_title="Probability")
fig.show()
# %%
