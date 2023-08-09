# %%
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

def mlp_effects_german(prompt, index, deactivate_context_hooks=[], activate_context_hooks=[]):
    """Customised to L5 and L8 context neurons"""
    downstream_components = [(f"blocks.{layer}.hook_{component}_out") for layer in [9, 10, 11] for component in ['mlp', 'attn']]
    
    original, ablated, direct_effect, _ = haystack_utils.get_direct_effect(
            prompt, model, pos=index, context_ablation_hooks=deactivate_context_hooks, context_activation_hooks=activate_context_hooks,
            deactivated_components=tuple(downstream_components), activated_components=("blocks.8.hook_mlp_out",))
    
    data = [original, ablated, direct_effect]
    for layer in [9, 10, 11]:
            _, _, _, activated_component_loss = haystack_utils.get_direct_effect(
                    prompt, model, pos=index, context_ablation_hooks=deactivate_context_hooks, context_activation_hooks=activate_context_hooks,
                    deactivated_components=tuple(component for component in downstream_components if component != f"blocks.{layer}.hook_mlp_out"),
                    activated_components=(f"blocks.{layer}.hook_mlp_out",))
            data.append(activated_component_loss)
    return data

def attn_effects_german(prompt, index, deactivate_context_hooks=[], activate_context_hooks=[]):
    """Customised to L5 and L8 context neurons"""
    downstream_components = [(f"blocks.{layer}.hook_{component}_out") for layer in [6, 7, 9, 10, 11] for component in ['mlp', 'attn']]

    data = []
    for layer in [9, 10, 11]:
            _, _, _, activated_component_loss = haystack_utils.get_direct_effect(
                    prompt, model, pos=index, context_ablation_hooks=deactivate_context_hooks, context_activation_hooks=activate_context_hooks,
                    deactivated_components=tuple(component for component in downstream_components if component != f"blocks.{layer}.hook_mlp_out"),
                    activated_components=(f"blocks.{layer}.hook_attn_out",))
            data.append(activated_component_loss)
    return data

def component_analysis(end_strings: list[str] | str, deactivate_context_hooks=[], activate_context_hooks=[]):
    if isinstance(end_strings, str):
        end_strings = [end_strings]
    for end_string in end_strings:
        print(model.to_str_tokens(end_string))
        random_prompts = haystack_utils.generate_random_prompts(end_string, model, common_tokens, 400, length=20)
        data = mlp_effects_german(random_prompts, -1, deactivate_context_hooks=deactivate_context_hooks, activate_context_hooks=activate_context_hooks)

        haystack_utils.plot_barplot([[item.cpu().flatten().mean().item()] for item in data],
                                        names=['activated', 'ablated', 'direct effect'] + [f'{i}{j}' for j in [9, 10, 11] for i in ["MLP"]], # + ["MLP9 + MLP11"]
                                        title=f'Loss increases from patching activated context neuron into various <br> MLP components and disabling elsewhere for end string \"{end_string}\"')
        
# %%

mid_word_prompt = " seinen Antworten"
new_word_prompt = " seine Antwort auf"

#%%
# Decrease loss
# component_analysis(mid_word_prompt, deactivate_context_hooks=snap_pos_to_peak_1_hook)
# %%
# Increase loss
component_analysis(mid_word_prompt, deactivate_context_hooks=snap_pos_to_peak_2_hook)
# %%
# Decrease loss
# component_analysis(new_word_prompt, deactivate_context_hooks=snap_pos_to_peak_2_hook)
# %%
# Increase loss
component_analysis(new_word_prompt, deactivate_context_hooks=snap_pos_to_peak_1_hook)
# %%

downstream_components = [(f"blocks.{layer}.hook_{component}_out") for layer in [9, 10, 11] for component in ['mlp', 'attn']]

# 1. Patch context to peak 1, evaluate on new word tokens

deactivate_context_hooks = snap_pos_to_peak_1_hook
activate_context_hooks = []
data = []
for prompt in tqdm(german_data[:200]):
    tokens = model.to_tokens(prompt)[0]
    mask = get_next_token_punctuation_mask(tokens)[:-1].cpu()

    # Snap to peak 1
    # Let other components read from normal context neuron
    # Direct = ablate all later components
    # _, _, direct_effect_p1, _ = haystack_utils.get_direct_effect(
    #         prompt, model, pos=None, context_ablation_hooks=snap_pos_to_peak_2_hook, context_activation_hooks=snap_pos_to_peak_1_hook,
    #         deactivated_components=tuple(downstream_components), activated_components=[f"blocks.{8}.hook_mlp_out"])
    
    # Indirect = activate all later components
    # Leave context neuron as is
    # Let later components read from snapped context neuron
    original_p1, ablated_p1, _, indirect_effect_p1 = haystack_utils.get_direct_effect(
            prompt, model, pos=None, context_ablation_hooks=snap_pos_to_peak_1_hook, context_activation_hooks=snap_pos_to_peak_2_hook,
            deactivated_components=[f"blocks.{8}.hook_mlp_out"], activated_components=downstream_components)
    _, _, _, direct_effect_p1 = haystack_utils.get_direct_effect(
        prompt, model, pos=None, context_ablation_hooks=snap_pos_to_peak_1_hook, context_activation_hooks=snap_pos_to_peak_2_hook,
        deactivated_components=downstream_components, activated_components=[f"blocks.{8}.hook_mlp_out"])
    
    # Check new word losses when snapping to peak 1
    original_p1 = original_p1[mask].tolist()
    ablated_p1 = ablated_p1[mask].tolist()
    direct_effect_p1 = direct_effect_p1[mask].tolist()#
    indirect_effect_p1 = indirect_effect_p1[mask].tolist()

    data.extend([["Original", loss] for loss in original_p1])
    data.extend([["Total P1 loss", loss] for loss in ablated_p1])
    data.extend([["Direct P1 loss", loss] for loss in direct_effect_p1])
    data.extend([["Indirect P1 loss", loss] for loss in indirect_effect_p1])

df = pd.DataFrame(data, columns=["Type", "Loss"])

# Group by 'Type' and calculate mean and 95% confidence interval
grouped_df = df.groupby('Type')['Loss'].agg(['mean', 'count', 'std'])
grouped_df['CI'] = grouped_df['std'] / grouped_df['count']**0.5 * stats.t.ppf((1 + 0.95) / 2, grouped_df['count'] - 1)

# Reset index to use 'Type' as a column
grouped_df.reset_index(inplace=True)

# Create the barplot
fig = px.bar(grouped_df, x="Type", y="mean", error_y="CI", title="Loss on new word tokens when snapping to peak 1")
fig.update_layout(
    yaxis=dict(range=[3.5, 4]),
    xaxis=dict(categoryorder='array', categoryarray=["Original", "Total P1 loss", "Direct P1 loss", "Indirect P1 loss"]),
    xaxis_title="",
    yaxis_title="Loss"
)
fig.show()
# %%


# %%
for prompt in german_data[:10]:
    # Snap to peak 2
    # Direct = ablate all later components
    original_p2, ablated_p2, direct_effect_p2, _ = haystack_utils.get_direct_effect(
            prompt, model, pos=None, context_ablation_hooks=snap_pos_to_peak_2_hook, context_activation_hooks=activate_context_hooks,
            deactivated_components=tuple(downstream_components), activated_components=[])
    
    # Indirect = activate all later components
    _, _, _, indirect_effect_p2 = haystack_utils.get_direct_effect(
            prompt, model, pos=None, context_ablation_hooks=snap_pos_to_peak_2_hook, context_activation_hooks=activate_context_hooks,
            deactivated_components=[], activated_components=(downstream_components))
    


# %%
# snap to peak 1, evaluate new word tokens
data = []
for prompt in tqdm(german_data[:200]):
    tokens = model.to_tokens(prompt)[0]
    mask = get_next_token_punctuation_mask(tokens)[:-1].cpu()

    original_p1, activated_p1, ablated_p1, direct_effect_p1, indirect_effect_p1 = haystack_utils.get_context_effect(prompt, model,
                            context_ablation_hooks=snap_pos_to_peak_1_hook, context_activation_hooks=snap_pos_to_peak_2_hook,
                            downstream_components=downstream_components, pos=None)

    # Check new word losses when snapping to peak 1
    original_p1 = original_p1.flatten()[mask].tolist()
    activated_p1 = activated_p1.flatten()[mask].tolist()
    ablated_p1 = ablated_p1.flatten()[mask].tolist()
    direct_effect_p1 = direct_effect_p1.flatten()[mask].tolist()#
    indirect_effect_p1 = indirect_effect_p1.flatten()[mask].tolist()

    data.extend([["Original", loss] for loss in original_p1])
    data.extend([["Activated P2", loss] for loss in activated_p1])
    data.extend([["Total P1", loss] for loss in ablated_p1])
    data.extend([["Direct P1", loss] for loss in direct_effect_p1])
    data.extend([["Indirect P1", loss] for loss in indirect_effect_p1])
    #break

df = pd.DataFrame(data, columns=["Type", "Loss"])

# Group by 'Type' and calculate mean and 95% confidence interval
grouped_df = df.groupby('Type')['Loss'].agg(['mean', 'count', 'std'])
grouped_df['CI'] = grouped_df['std'] / grouped_df['count']**0.5 * stats.t.ppf((1 + 0.95) / 2, grouped_df['count'] - 1)

# Reset index to use 'Type' as a column
grouped_df.reset_index(inplace=True)

# Create the barplot
fig = px.bar(grouped_df, x="Type", y="mean", error_y="CI", title="Loss on new word tokens when snapping to peak 1")
fig.update_layout(
    yaxis=dict(range=[3.5, 4]),
    xaxis=dict(categoryorder='array', categoryarray=["Original", "Activated P2", "Direct P1", "Indirect P1", "Total P1"]),
    xaxis_title="",
    yaxis_title="Loss"
)
fig.show()
# %%
# Snap to Peak 2, evaluate continuation token loss
data = []
for prompt in tqdm(german_data[:200]):
    tokens = model.to_tokens(prompt)[0]
    mask = get_next_token_punctuation_mask(tokens)[:-1].cpu()

    original_p1, activated_p1, ablated_p1, direct_effect_p1, indirect_effect_p1 = haystack_utils.get_context_effect(prompt, model,
                            context_ablation_hooks=snap_pos_to_peak_2_hook, context_activation_hooks=snap_pos_to_peak_1_hook,
                            downstream_components=downstream_components, pos=None)

    # Check new word losses when snapping to peak 1
    original_p1 = original_p1.flatten()[~mask].tolist()
    activated_p1 = activated_p1.flatten()[~mask].tolist()
    ablated_p1 = ablated_p1.flatten()[~mask].tolist()
    direct_effect_p1 = direct_effect_p1.flatten()[~mask].tolist()#
    indirect_effect_p1 = indirect_effect_p1.flatten()[~mask].tolist()

    data.extend([["Original", loss] for loss in original_p1])
    data.extend([["Activated P1", loss] for loss in activated_p1])
    data.extend([["Total P2", loss] for loss in ablated_p1])
    data.extend([["Direct P2", loss] for loss in direct_effect_p1])
    data.extend([["Indirect P2", loss] for loss in indirect_effect_p1])
    #break

df = pd.DataFrame(data, columns=["Type", "Loss"])

# Group by 'Type' and calculate mean and 95% confidence interval
grouped_df = df.groupby('Type')['Loss'].agg(['mean', 'count', 'std'])
grouped_df['CI'] = grouped_df['std'] / grouped_df['count']**0.5 * stats.t.ppf((1 + 0.95) / 2, grouped_df['count'] - 1)

# Reset index to use 'Type' as a column
grouped_df.reset_index(inplace=True)

# Create the barplot
fig = px.bar(grouped_df, x="Type", y="mean", error_y="CI", title="Loss on continuation tokens when snapping to peak 2")
fig.update_layout(
    yaxis=dict(range=[1, 2]),
    xaxis=dict(categoryorder='array', categoryarray=["Original", "Activated P1", "Direct P2", "Indirect P2", "Total P2"]),
    xaxis_title="",
    yaxis_title="Loss"
)
fig.show()
# %%
