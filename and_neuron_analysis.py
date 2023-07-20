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

prompt_tuple = haystack_utils.get_trigram_prompts(prompts, continuation_tokens, continuation_tokens)
prev_token_direction, curr_token_direction = haystack_utils.get_residual_trigram_directions(prompt_tuple, model, 4)
# %%
context_direction = model.W_out[3, 669, :]

# %%
def get_cosine_sim(direction: Float[Tensor, "d_res"], layer=5) -> Float[Tensor, "d_mlp"]:
    cosine = torch.nn.CosineSimilarity(dim=1)
    return cosine(model.W_in[layer].T, direction.unsqueeze(0))

prev_token_sim = get_cosine_sim(prev_token_direction, 5)
curr_token_sim = get_cosine_sim(curr_token_direction, 5)
context_sim = get_cosine_sim(context_direction, 5)
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

#plot_histogram(prev_token_sim, curr_token_sim, context_sim, "Previous Token", "Current Token", "Context")
# %%

all = haystack_utils.union_where([prev_token_sim, curr_token_sim, context_sim], 0.05)
print(all)

for neuron in all:
    print(prev_token_sim[neuron], curr_token_sim[neuron], context_sim[neuron])
# %%

df = haystack_utils.get_trigram_neuron_activations(prompt_tuple, model, deactivate_neurons_fwd_hooks ,5)

# %%

df["PrevTokenSim"] = prev_token_sim.cpu().numpy()
df["CurrTokenSim"] = curr_token_sim.cpu().numpy()
df["ContextSim"] = context_sim.cpu().numpy()
df["AllSim"] = df.index.isin(all.tolist())

# %%
df["diff"] = df["YYY"] - df["YYN"]
df["ContextAnd"] = (df["NNY"] < 0) & (df["YYY"]>0)

#pivot_df[pivot_df.index.isin(neurons.tolist())]
# %%

def compute_loss_increase(prompts, df, neurons, ablate_mode="NNN", layer=5):
    
    mean_activations = torch.Tensor(df[df.index.isin(neurons.tolist())][ablate_mode].tolist()).cuda()

    def ablate_mlp_hook(value, hook):
        value[:, :, neurons] = mean_activations
        return value

    loss = model(prompts, return_type="loss", loss_per_token=True)[:, -1].mean().item()
    with model.hooks(fwd_hooks=[(f"blocks.{layer}.mlp.hook_pre", ablate_mlp_hook)]):
        ablated_loss = model(prompts, return_type="loss", loss_per_token=True)[:, -1].mean().item()

    return loss, ablated_loss
# %%
# High loss diff neurons
high_loss_diff_neurons = torch.LongTensor([84, 178, 213, 255, 395, 905, 1250, 1510, 1709]).cuda()

#neurons = torch.LongTensor(df[:19].index.tolist()).cuda()

and_neurons = torch.LongTensor(df[df["AND"]].index.tolist()).cuda()

print(compute_loss_increase(prompts, df, high_loss_diff_neurons, ablate_mode="NNN"))
print(compute_loss_increase(prompts, df, and_neurons, ablate_mode="NNN"))
# %%
