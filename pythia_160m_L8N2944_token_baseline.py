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
