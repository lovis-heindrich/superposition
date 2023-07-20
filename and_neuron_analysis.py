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

activate_neurons_fwd_hooks, deactivate_neurons_fwd_hooks = haystack_utils.get_context_ablation_hooks(3, [669], model)
all_ignore, _ = haystack_utils.get_weird_tokens(model, plot_norms=False)

german_data = haystack_utils.load_json_data("data/german_europarl.json")[:200]
common_tokens = haystack_utils.get_common_tokens(german_data, model, all_ignore, k=100)

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

context_direction = model.W_out[3, 669, :]

def get_cosine_sim(direction: Float[Tensor, "d_res"], layer=5) -> Float[Tensor, "d_mlp"]:
    cosine = torch.nn.CosineSimilarity(dim=1)
    return cosine(model.W_in[layer].T, direction.unsqueeze(0))

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

def compute_mlp_loss(prompts, df, neurons, ablate_mode="NNN", layer=5, compute_original_loss=False):

    mean_activations = torch.Tensor(df[df.index.isin(neurons.tolist())][ablate_mode].tolist()).cuda()

    def ablate_mlp_hook(value, hook):
        value[:, :, neurons] = mean_activations
        return value

    with model.hooks(fwd_hooks=[(f"blocks.{layer}.mlp.hook_pre", ablate_mlp_hook)]):
        ablated_loss = model(prompts, return_type="loss", loss_per_token=True)[:, -1].mean().item()

    if compute_original_loss:
        loss = model(prompts, return_type="loss", loss_per_token=True)[:, -1].mean().item()
        return loss, ablated_loss
    return ablated_loss
# %%

def process_data_frame(ngram:str):
    prompts = haystack_utils.generate_random_prompts(ngram, model, common_tokens, 400, length=20)
    if ngram.startswith(" "):
        prompt_tuple = haystack_utils.get_trigram_prompts(prompts, new_word_tokens, continuation_tokens)
    else:
        prompt_tuple = haystack_utils.get_trigram_prompts(prompts, continuation_tokens, continuation_tokens)
    prev_token_direction, curr_token_direction = haystack_utils.get_residual_trigram_directions(prompt_tuple, model, 4)

    prev_token_sim = get_cosine_sim(prev_token_direction, 5)
    curr_token_sim = get_cosine_sim(curr_token_direction, 5)
    context_sim = get_cosine_sim(context_direction, 5)

    all = haystack_utils.union_where([prev_token_sim, curr_token_sim, context_sim], 0.05)

    df = haystack_utils.get_trigram_neuron_activations(prompt_tuple, model, deactivate_neurons_fwd_hooks ,5)

    df["PrevTokenSim"] = prev_token_sim.cpu().numpy()
    df["CurrTokenSim"] = curr_token_sim.cpu().numpy()
    df["ContextSim"] = context_sim.cpu().numpy()
    df["AllSim"] = df.index.isin(all.tolist())

    df["AblationDiff"] = df["YYY"] - df["YYN"]
    df["And"] = (df["YYY"]>0) & (df["YYN"]<=0) & (df["YNY"]<=0) & (df["NYY"]<=0) & (df["YNN"]<=0) & (df["NNY"]<=0)& (df["NYN"]<=0)
    df["NegAnd"] = (df["YYY"]<=0) & (df["YYN"]>0) & (df["YNY"]>0) & (df["NYY"]>0) & (df["YNN"]>0) & (df["NNY"]>0)& (df["NYN"]>0)


    original_loss, ablated_loss = compute_mlp_loss(prompts, df, torch.LongTensor([i for i in range(model.cfg.d_mlp)]).cuda(), compute_original_loss=True)
    print(original_loss, ablated_loss)

    ablated_losses = []
    for neuron in tqdm(range(model.cfg.d_mlp)):
        ablated_loss = compute_mlp_loss(prompts, df, torch.LongTensor([neuron]).cuda(), ablate_mode="NNN")
        ablated_losses.append(ablated_loss)

    df["FullAblationLossIncrease"] = ablated_losses
    df["FullAblationLossIncrease"] -= original_loss

    # Neuron wise loss increase when ablating context neuron
    original_loss, ablated_loss = compute_mlp_loss(prompts, df, torch.LongTensor([i for i in range(model.cfg.d_mlp)]).cuda(), compute_original_loss=True, ablate_mode="YYN")
    print(original_loss, ablated_loss)

    ablated_losses = []
    for neuron in tqdm(range(model.cfg.d_mlp)):
        ablated_loss = compute_mlp_loss(prompts, df, torch.LongTensor([neuron]).cuda(), ablate_mode="YYN")
        ablated_losses.append(ablated_loss)

    df["ContextAblationLossIncrease"] = ablated_losses
    df["ContextAblationLossIncrease"] -= original_loss
    return df

# %%

options = ["orschlägen", " häufig", " beweglich"]
option_dfs = []
for ngram in options:
    option_dfs.append(process_data_frame(ngram))

# %%
for option, df in zip(options, option_dfs):
    df.to_pickle(f"data/and_neurons/df_{option.strip()}.pkl")

# %%
index = 0
ngram = options[index]
df = option_dfs[index]
df[df["And"]]

# %%
# Compute losses for sets of neurons

# # High loss diff neurons
# high_loss_diff_neurons = torch.LongTensor([84, 178, 213, 255, 395, 905, 1250, 1510, 1709]).cuda()

# #neurons = torch.LongTensor(df[:19].index.tolist()).cuda()

and_neurons = torch.LongTensor(df[df["And"]].index.tolist()).cuda()

prompts = haystack_utils.generate_random_prompts(ngram, model, common_tokens, 2000, length=20)

#print(compute_mlp_loss(prompts, df, high_loss_diff_neurons, ablate_mode="NNN"))
print(compute_mlp_loss(prompts, df, and_neurons, ablate_mode="YYN", compute_original_loss=True))
# %%
len(common_tokens)
# %%
