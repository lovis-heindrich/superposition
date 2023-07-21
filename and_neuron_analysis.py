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
import json

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

def process_data_frame(ngram:str, batch=400):
    prompts = haystack_utils.generate_random_prompts(ngram, model, common_tokens, batch, length=20)
    if ngram.startswith(" "):
        prompt_tuple = haystack_utils.get_trigram_prompts(prompts, new_word_tokens, continuation_tokens)
    else:
        prompt_tuple = haystack_utils.get_trigram_prompts(prompts, continuation_tokens, continuation_tokens)
    prev_token_direction, curr_token_direction = haystack_utils.get_residual_trigram_directions(prompt_tuple, model, 4)

    prev_token_sim = get_cosine_sim(prev_token_direction, 5)
    curr_token_sim = get_cosine_sim(curr_token_direction, 5)
    context_sim = get_cosine_sim(context_direction, 5)

    pos_sim = haystack_utils.union_where([prev_token_sim, curr_token_sim, context_sim], 0.02)
    neg_sim = haystack_utils.union_where([prev_token_sim, curr_token_sim, context_sim], 0.02, greater=False)

    df = haystack_utils.get_trigram_neuron_activations(prompt_tuple, model, deactivate_neurons_fwd_hooks ,5)

    df["PrevTokenSim"] = prev_token_sim.cpu().numpy()
    df["CurrTokenSim"] = curr_token_sim.cpu().numpy()
    df["ContextSim"] = context_sim.cpu().numpy()
    df["PosSim"] = df.index.isin(pos_sim.tolist())
    df["NegSim"] = df.index.isin(neg_sim.tolist())

    df["AblationDiff"] = df["YYY"] - df["YYN"]
    df["And"] = (df["YYY"]>0) & (df["YYN"]<=0) & (df["YNY"]<=0) & (df["NYY"]<=0) & (df["YNN"]<=0) & (df["NNY"]<=0)& (df["NYN"]<=0)
    df["NegAnd"] = (df["YYY"]<=0) & (df["YYN"]>0) & (df["YNY"]>0) & (df["NYY"]>0) & (df["YNN"]>0) & (df["NNY"]>0)& (df["NYN"]>0)

    # Look for neurons that consistently respond to all 3 directions
    df["Boosted"] = (df["YNN"]>df["NNN"])&(df["NYN"]>df["NNN"])&(df["NNY"]>df["NNN"])&\
                    (df["YYY"]>df["YNN"])&(df["YYY"]>df["NYN"])&(df["YYY"]>df["NNY"])&\
                    (df["YYY"]>0) # Negative boosts don't matter

    df["Deboosted"] = (df["YNN"]<df["NNN"])&(df["NYN"]<df["NNN"])&(df["NNY"]<df["NNN"])&\
                    (df["YYY"]<df["YNN"])&(df["YYY"]<df["NYN"])&(df["YYY"]<df["NNY"])&\
                    (df["NNN"]>0) # Deboosting negative things doesn't matter

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
# Add loss increases for various sets of neurons

all_losses = {}
for option, df in tqdm(zip(options, option_dfs)):
    all_losses[option] = {}
    prompts = haystack_utils.generate_random_prompts(option, model, common_tokens, 1000, length=20)

    for ablation_mode in ["YYN", "NNN"]:
        all_losses[option][ablation_mode] = {}
        original_loss, ablated_loss = compute_mlp_loss(prompts, df, torch.LongTensor([i for i in range(model.cfg.d_mlp)]).cuda(), compute_original_loss=True, ablate_mode=ablation_mode)

        if ablation_mode == "YYN":
            high_loss_neurons = df.sort_values("ContextAblationLossIncrease", ascending=False)[:10].index.tolist()
            print(high_loss_neurons)
            #print(df.loc[high_loss_neurons]["ContextAblationLossIncrease"])
        else:
            high_loss_neurons = df.sort_values("FullAblationLossIncrease", ascending=False)[:10].index.tolist()
        top_neuron_loss = compute_mlp_loss(prompts, df, torch.LongTensor(high_loss_neurons).cuda(), ablate_mode=ablation_mode)
        print(top_neuron_loss)
        print(model.to_str_tokens(prompts[0]))
        all_losses[option][ablation_mode]["Original"] = original_loss
        all_losses[option][ablation_mode]["Ablated"] = ablated_loss
        all_losses[option][ablation_mode]["TopNeurons"] = top_neuron_loss

        pos_and_neurons = torch.LongTensor(df[df["And"]].index.tolist()).cuda()
        neg_and_neurons = torch.LongTensor(df[df["NegAnd"]].index.tolist()).cuda()
        both_and_neurons = torch.LongTensor(df[df["And"] | df["NegAnd"]].index.tolist()).cuda()
        
        pos_boost_neurons = torch.LongTensor(df[df["Boosted"]].index.tolist()).cuda()
        neg_boost_neurons = torch.LongTensor(df[df["Deboosted"]].index.tolist()).cuda()
        both_boost_neurons = torch.LongTensor(df[df["Boosted"] | df["Deboosted"]].index.tolist()).cuda()
        
        pos_sim_neurons = torch.LongTensor(df[df["PosSim"]].index.tolist()).cuda()
        neg_sim_neurons = torch.LongTensor(df[df["NegSim"]].index.tolist()).cuda()
        both_sim_neurons = torch.LongTensor(df[df["PosSim"] | df["NegSim"]].index.tolist()).cuda()

        neuron_sets = [pos_and_neurons, neg_and_neurons, both_and_neurons, 
                       pos_boost_neurons, neg_boost_neurons, both_boost_neurons, 
                       pos_sim_neurons, neg_sim_neurons, both_sim_neurons]
        
        neuron_set_names = ["PositiveAND", "NegativeAND", "BothAND",
                            "PositiveBoost", "NegativeBoost", "BothBoost",
                            "PositiveSim", "NegativeSim", "BothSim"]
        
        if "NumNeurons" not in all_losses[option].keys():
            all_losses[option]["NumNeurons"] = {
                "Original": 0,
                "Ablated": model.cfg.d_mlp,
                "TopNeurons": len(high_loss_neurons),
            }
            for i in range(len(neuron_sets)):
                all_losses[option]["NumNeurons"][neuron_set_names[i]] = len(neuron_sets[i])
        for name, neurons in zip(neuron_set_names, neuron_sets):
            all_losses[option][ablation_mode][name] = compute_mlp_loss(prompts, df, neurons, ablate_mode=ablation_mode)
# %%
with open("data/and_neurons/set_losses.json", "w") as f:
    json.dump(all_losses, f)

# %%

option = "orschlägen"
ablation_mode = "YYN"
prompts = haystack_utils.generate_random_prompts(option, model, common_tokens, 1000, length=20)

names = list(all_losses[option][ablation_mode].keys())
losses = [[all_losses[option][ablation_mode][name]] for name in names]

print(len(names), len(losses))
print([len(x) for x in losses])
haystack_utils.plot_barplot(losses, names)

# %%

# Next steps
# Check if there are neurons with high similarity for the token directions in earlier layers
# Check if ablating them increases loss

# %% 
df["ContextAblationLossIncrease"].mean()

# %%

weird_neurons = [84, 905, 1709, 1747, 1510, 1765, 1868, 627, 297, 1232]
#weird_neurons = [297, 627, 905,  1709,  1747, 84, 1765, 1868, 1232,1510] # 
compute_mlp_loss(prompts, df, torch.LongTensor(weird_neurons).cuda(), ablate_mode="YYN")

original_loss, ablated_loss = compute_mlp_loss(prompts, df, torch.LongTensor([i for i in range(model.cfg.d_mlp)]).cuda(), compute_original_loss=True, ablate_mode="YYN")
print(original_loss, ablated_loss)

# %%

with model.hooks(fwd_hooks=deactivate_neurons_fwd_hooks):
    ablated_loss, ablated_cache = model.run_with_cache(prompts, return_type="loss", loss_per_token=True)[:, -1].mean().item()
    print("Context ablated", ablated_loss)

def deactivate_neurons_hook(value, hook):
        value[:, :, weird_neurons] = ablated_cache[hook.name][:, :, weird_neurons].mean((0, 1))
        return value

deactivate_weird_neurons_hook = [("blocks.5.mlp.hook_post", deactivate_neurons_hook)]

with model.hooks(fwd_hooks=deactivate_neurons_fwd_hooks+deactivate_weird_neurons_hook):
    ablated_loss, ablated_cache = model.run_with_cache(prompts, return_type="loss", loss_per_token=True)[:, -1].mean().item()
    print("Ablated + Patched", ablated_loss)
# %%

prompts.shape
model.to_str_tokens(prompts[0])
# %%
