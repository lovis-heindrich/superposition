import warnings
from hook_utils import save_activation
from transformer_lens import HookedTransformer, ActivationCache, utils
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float, Int
from torch import Tensor
import einops
from tqdm.auto import tqdm
import torch
from typing import Callable, List, Tuple, Literal
import plotly.express as px
import plotly.graph_objects as go
import gc
import numpy as np
from einops import einsum
from IPython.display import display, HTML
import re
from pathlib import Path
import json
import pandas as pd


def DLA(prompts: List[str], model: HookedTransformer) -> tuple[Float[Tensor, "component"], list[str]]:
    logit_attributions = []
    for prompt in tqdm(prompts):
        tokens = model.to_tokens(prompt)
        answers = tokens[:, 1:]
        tokens = tokens[:, :-1]
        answer_residual_directions = model.tokens_to_residual_directions(answers)  # [batch pos d_model]
        _, cache = model.run_with_cache(tokens)
        accumulated_residual, labels = cache.accumulated_resid(layer=-1, incl_mid=False, pos_slice=None, return_labels=True)
        scaled_residual_stack = cache.apply_ln_to_stack(accumulated_residual, layer = -1, pos_slice=None)
        logit_attribution = einsum(scaled_residual_stack, answer_residual_directions, "component batch pos d_model, batch pos d_model -> component") / answers.shape[1]
        logit_attributions.append(logit_attribution)
    
    logit_attributions = torch.stack(logit_attributions)
    return logit_attributions, labels


def get_mlp_activations(
    prompts: List[str],
    layer: int,
    model: HookedTransformer,
    num_prompts: int = -1,
    context_crop_start=10,
    context_crop_end=400,
    mean=True,
    hook_pre = False,
    pos=None,
    neurons: Int[Tensor, "n_neurons"] | None = None,
) -> Float[Tensor, "num_activations d_mlp"]:
    """Runs the model through a list of prompts and stores the mlp activations for a given layer. Might be slow for large batches as examples are run one by one.

    Args:
        prompts (list[str]): Prompts to run through the model.
        layer (int): Layer to extract activations from.
        model (HookedTransformer): Model to run.
        num_prompts (int, optional): Number of prompts to run. Defaults to -1 (all prompts).
        context_crop_start (int, optional): Crops the start context position to avoid low neuron activations. Defaults to 10.
        context_crop_end (int, optional): Crops the end context position to avoid low neuron activations. Defaults to 400.

    Returns:
        Float[Tensor, "num_activations d_mlp"]: Stacked activations for each prompt and position.
    """
    acts = []
    if mean:
        act_lens = []

    if hook_pre:
        act_label = f"blocks.{layer}.mlp.hook_pre"
    else:
        act_label = f"blocks.{layer}.mlp.hook_post"
    neurons = neurons or torch.arange(model.cfg.d_mlp)
    if num_prompts == -1:
        num_prompts = len(prompts)

    for i in tqdm(range(num_prompts)):
        tokens = model.to_tokens(prompts[i])
        with model.hooks([(act_label, save_activation)]):
            model.run_with_hooks(tokens)
            act = model.hook_dict[act_label].ctx['activation'][:, context_crop_start:context_crop_end, :]
        if pos is not None:
            act = act[:, pos, neurons].unsqueeze(1)
        act = einops.rearrange(act, "batch pos d_mlp -> (batch pos) d_mlp")
        if mean:
            act_lens.append(tokens.shape[1])
            act = torch.mean(act, dim=0)
        acts.append(act)
    if mean:
        sum_act_lens = sum(act_lens)
        weighted_mean = torch.sum(torch.stack([a*b/sum_act_lens for a, b in zip(acts, act_lens)]), dim=0)
        return weighted_mean
    acts = torch.concat(acts, dim=0)
    return acts


def weighted_mean(mean_acts: list[Float[Tensor, "n_acts n_neurons"]], batch_sizes: list[int]):
    """global mean of means, determined using the batch size used to calculate each mean"""
    sum_act_lens = sum(batch_sizes)
    normalized_means = [a*b/sum_act_lens for a, b in zip(mean_acts, batch_sizes)]
    weighted_mean = torch.sum(torch.stack(normalized_means), dim=0)
    return weighted_mean


def get_average_loss(data: List[str], model: HookedTransformer, crop_context=-1, fwd_hooks=[], positionwise=False):
    """
    Mean over all tokens in the data, not the mean of the mean of each batch. 
    Uses a mask to account for padding tokens, differing prompt lengths, and final tokens not having a loss value.
    """
    if crop_context == -1:
        crop_context = model.cfg.n_ctx

    position_counts = torch.zeros(model.cfg.n_ctx).cuda()
    position_loss = torch.zeros(model.cfg.n_ctx).cuda()

    for item in data:
        tokens = model.to_tokens(item)[:, :crop_context].cuda()
        loss = model.run_with_hooks(tokens, return_type="loss", loss_per_token=True, fwd_hooks=fwd_hooks) # includes BOS, excludes final token of longest prompt in batch

        # Produce a mask of every token which we expect to produce a valid loss value. Excludes padding tokens and the final token of each row.
        active_tokens = tokens[:, :] != model.tokenizer.pad_token_id
        active_tokens[:, 0] = True  # BOS token ID is sometimes equivalent to pad token ID so we manually set it to true
        # Set each final token prediction to False
        last_true_token_mask = (active_tokens.flip(dims=[1]).cumsum(dim=1) == 1).flip(dims=[1])
        last_true_indices = last_true_token_mask.nonzero() # Tensor of tuples of each final True index in the batch
        active_tokens[last_true_indices[:, 0], last_true_indices[:, 1]] = False
        # Remove the final column which will never have an associated loss. The mask size is now equal to the loss tensor size.
        active_tokens = active_tokens[:, :-1] 
        # Set loss of inactive tokens to 0
        inactive_tokens = torch.logical_not(active_tokens)
        loss[inactive_tokens] = 0  # [batch, pos]

        # Sum loss batch-wise and update positionwise loss.
        position_loss[0:loss.shape[1]] += loss.sum(dim=0)
        # Increment by the number of non-pad tokens at each position.
        position_counts[0:active_tokens.shape[1]] += active_tokens.sum(dim=0).float()

    if positionwise:
        avg_position_loss = []
        for i in range(len(position_loss)):
            avg_position_loss.append((position_loss[i] / position_counts[i]).item() if position_counts[i] != 0 else 0)
        return avg_position_loss

    return position_loss.sum() / position_counts.sum()


def get_direct_loss_increase_for_component(
        prompts: List[str],
        model: HookedTransformer,
        fwd_hooks=[],
        patched_component=8,
        crop_context_end: None | int=None,
        disable_progress_bar=False
    ):
    """
    Measures the direct effect on loss of patching the component.
    - Does not include the effect of the patched component on other components.
    - Does include the effect of the pre-patched component on other components.
    - Does include the contribution of the pre-patched component to the layer normalizations of the residual stream.

    Get the original loss of a forward pass. Patch in the contribution to the residual of the patched component and remove 
    the contribution of the original component, then measure the patched loss. Return the patched and original losses.

    Print the patched loss, original loss, and patched loss increase as a percentage of the original loss.
    """
    original_losses = []
    patched_losses = []
    for prompt in tqdm(prompts, disable=disable_progress_bar):
        if crop_context_end is not None:
            tokens = model.to_tokens(prompt)[:, :crop_context_end]
        else:
            tokens = model.to_tokens(prompt)

        original_loss, _, original_cache, ablated_cache = get_caches_single_prompt(
            prompt, model, fwd_hooks=fwd_hooks, crop_context_end=crop_context_end, return_type="loss")

        # Applying layer norm here with layer=-1 would apply the final layer's layer norm to the component output we're going to patch.
        # Since we're adding/removing these components from hook_resid_post before the final layer norm is appled, we don't need this.
        original_per_layer_residual, original_labels = original_cache.decompose_resid(layer=-1, return_labels=True, apply_ln=False)  # [component batch pos residual]
        ablated_per_layer_residual, ablated_labels = ablated_cache.decompose_resid(layer=-1, return_labels=True, apply_ln=False)

        # ['embed', '0_attn_out', '0_mlp_out', '1_attn_out', '1_mlp_out', '2_attn_out', '2_mlp_out', '3_attn_out', '3_mlp_out', '4_attn_out', '4_mlp_out', '5_attn_out', '5_mlp_out']
        # Ablate at the final residual stream value to remove the direct component output
        def swap_cache_hook(value, hook):
            # Batch, pos, residual
            value -= original_per_layer_residual[patched_component]
            value += ablated_per_layer_residual[patched_component]
        
        with model.hooks(fwd_hooks=[(f'blocks.5.hook_resid_post', swap_cache_hook)]):
            patched_loss = model(tokens, return_type="loss")

        original_losses.append(original_loss)
        patched_losses.append(patched_loss.item())


    print(f"Original loss: {np.mean(original_losses):.2f}, patched loss: {np.mean(patched_losses):.2f} (+{((np.mean(patched_losses) - np.mean(original_losses)) / np.mean(original_losses))*100:.2f}%)")
    return np.mean(original_losses), np.mean(patched_losses)


def get_ablated_performance(data: List[str], model: HookedTransformer, fwd_hooks: List[Tuple]=[], batch_size=1, display_tqdm=True):
    assert batch_size == 1, "Only tested with batch size 1"

    original_losses = []
    ablated_losses = []
    for example in tqdm(data, disable=(not display_tqdm)):
        tokens = model.to_tokens(example)

        original_loss = model(tokens, return_type="loss")
        ablated_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=fwd_hooks)

        original_losses.append(original_loss.item())
        ablated_losses.append(ablated_loss.item())

    mean_original_loss = np.mean(original_losses)
    mean_ablated_loss = np.mean(ablated_losses)
    percent_increase = ((mean_ablated_loss - mean_original_loss) / mean_original_loss) * 100
    return mean_original_loss, mean_ablated_loss, percent_increase


def get_caches_single_prompt(
    prompt: str, 
    model: HookedTransformer, 
    fwd_hooks=[], 
    crop_context_end=None,
    return_type: str = "loss"
) -> Tuple[float, float, ActivationCache, ActivationCache]:
    """ Runs the model with and without ablation on a single prompt and returns the caches.

    Args:
        prompt (str): Prompt to run.
        model (HookedTransformer): Model to run.
        fwd_hooks (list, optional): Forward hooks to apply during ablation.
        crop_context_end (int, optional): Crops the tokens to the specified length.

    Returns:
        tuple[float, float, ActivationCache, ActivationCache]: original_loss, ablated_loss, original_cache, ablated_cache.
    """
    assert return_type in ["loss", "logits"]

    if crop_context_end is not None:
        tokens = model.to_tokens(prompt)[:, :crop_context_end]
    else:
        tokens = model.to_tokens(prompt)
  
    original_return_value, original_cache = model.run_with_cache(tokens, return_type=return_type)
    with model.hooks(fwd_hooks=fwd_hooks):
        ablated_return_value, ablated_cache = model.run_with_cache(tokens, return_type=return_type)
    
    if return_type == "loss":  
        return original_return_value.item(), ablated_return_value.item(), original_cache, ablated_cache
    
    return original_return_value, ablated_return_value, original_cache, ablated_cache


def generate_text(prompt, model, fwd_hooks=[], k=20, truncate_index=None):
    """
    Generate k tokens of text from the model given a tokenized prompt.
    Index into tokens up to the truncate index.
    Only tested with a batch size of 1.
    """
    tokens = model.to_tokens(prompt)
    tokens = tokens.cuda()
    
    if truncate_index is not None:
        tokens = tokens[:, :truncate_index]
    truncated_prompt = model.to_string(tokens[:, 1:]) # don't print BOS token

    with model.hooks(fwd_hooks=fwd_hooks):
        predictions = []
        logits = model(tokens)
        next_token = logits[0, -1].argmax(dim=-1)
        next_char = model.to_string(next_token)
        predictions.append(next_char)

        for _ in range(1, k):
            tokens = torch.cat([tokens, next_token[None, None]], dim=-1)
            logits = model(tokens)
            next_token = logits[0, -1].argmax(dim=-1)
            next_char = model.to_string(next_token)
            predictions.append(next_char)

        return "".join(truncated_prompt) + "".join(predictions)


def two_histogram(data_1: Float[Tensor, "n"], data_2: Float[Tensor, "n"], data_1_name="", data_2_name="", title: str = "", x_label: str= "", y_label: str="", histnorm="", n_bins=100):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data_1.cpu().numpy(), histnorm='percent', name=data_1_name))
    fig.add_trace(go.Histogram(x=data_2.cpu().numpy(), histnorm='percent', name=data_2_name))

    fig.update_layout(
        width=800,
        title_text=title, # title of plot
        xaxis_title_text=x_label, # xaxis label
        yaxis_title_text=y_label, # yaxis label
        bargap=0.2, # gap between bars of adjacent location coordinates
        bargroupgap=0.1 # gap between bars of the same location coordinates
    )
    fig.show()


def imshow(tensor, xaxis="", yaxis="", title="", **kwargs):
    plot_kwargs = {
        "color_continuous_scale":"RdBu", 
        "color_continuous_midpoint":0.0,
        "labels":{"x":xaxis, "y":yaxis}
    }
    plot_kwargs.update(kwargs)
    px.imshow(tensor, **plot_kwargs, title=title).show()


def line(x, xlabel="", ylabel="", title="", xticks=None, width=800, hover_data=None, show_legend=True, plot=True):
    
    # Avoid empty plot when x contains a single element
    if len(x) > 1:
        fig = px.line(x, title=title)
    else:
        fig = px.scatter(x, title=title)

    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, width=width, showlegend=show_legend)
    if xticks != None:
        fig.update_layout(
            xaxis = dict(
            tickmode = 'array',
            tickvals = [i for i in range(len(xticks))],
            ticktext = xticks,
            range=[-0.2, len(xticks)-0.8] 
            ),
        )
    
    #fig.update_yaxes(range=[3.45, 3.85])
    if hover_data != None:
        fig.update(data=[{'customdata': hover_data, 'hovertemplate': "Loss: %{y:.4f} (+%{customdata:.2f}%)"}])
    if plot:
        fig.show()
    else:
        return fig


def clean_cache():
    """Cleans the cache and empties the GPU cache.
    """
    gc.collect()
    torch.cuda.empty_cache()


def load_txt_data(path: str) -> List[str]:
    """Loads line separated dataset examples from a text file.

    Args:
        path (str): Path to the text file.

    Returns:
        list[str]: List of examples.
    """
    with open(Path(path), "r") as f:
        data = f.read().split("\n")
    min_len = min([len(example) for example in data])
    max_len = max([len(example) for example in data])
    print(
        f"{path}: Loaded {len(data)} examples with {min_len} to {max_len} characters each."
    )
    return data

def load_json_data(path: str) -> list[str]:
    with open(path, 'r') as f:
        data = json.load(f)

    min_len = min([len(example) for example in data])
    max_len = max([len(example) for example in data])
    print(
        f"{path}: Loaded {len(data)} examples with {min_len} to {max_len} characters each."
    )
    return data


def print_strings_as_html(strings: list[str], color_values: list[float], max_value: float=None, original_log_probs: list[float]=None, ablated_log_probs: list[float]=None, logit_difference: list[float]=None):
    """ Magic GPT function that prints a string as HTML and colors it according to a list of color values. Color values are normalized to the max value preserving the sign.
    """

    def normalize(values, max_value=None, min_value=None):
        if max_value is None:
            max_value = max(values)
        if min_value is None:
            min_value = min(values)
        min_value = abs(min_value)
        normalized = [value / max_value if value > 0 else value / min_value for value in values]
        return normalized
    
    if (original_log_probs is not None) and (ablated_log_probs is not None):
        assert len(strings) == len(color_values) == len(original_log_probs) == len(ablated_log_probs), f"Lengths of strings, color_values, original_log_probs, and ablated_log_probs must be equal. Got {len(strings)}, {len(color_values)}, {len(original_log_probs)}, and {len(ablated_log_probs)}."

    html = "<div>"
    
    # Normalize color values
    normalized_values = normalize(color_values, max_value, max_value)

    for i in range(len(strings)):
        color = color_values[i]
        normalized_color = normalized_values[i]
        
        if color < 0:
            red = int(max(0, min(255, (1 + normalized_color) * 255)))
            green = int(max(0, min(255, (1 + normalized_color) * 255)))
            blue = 255
        else:
            red = 255
            green = int(max(0, min(255, (1 - normalized_color) * 255)))
            blue = int(max(0, min(255, (1 - normalized_color) * 255)))
        
        # Calculate luminance to determine if text should be black
        luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255
        
        # Determine text color based on background luminance
        text_color = "black" if luminance > 0.5 else "white"

        #visible_string = re.sub(r'\s+', '&nbsp;', strings[i])
        visible_string = re.sub(r'\s+', '_', strings[i])
        
        
        if (original_log_probs is not None) and (ablated_log_probs is not None) and (logit_difference is not None):
            html += f'<span style="background-color: rgb({red}, {green}, {blue}); color: {text_color}; padding: 2px;" ' \
                    f'title="Difference: {color_values[i]:.4f}, Original logprob: {original_log_probs[i]:.4f}, Ablated logprob: {ablated_log_probs[i]:.4f}, Logit difference: {logit_difference[i]:.4f}">{visible_string}</span> '
        elif (original_log_probs is not None) and (ablated_log_probs is not None):
            html += f'<span style="background-color: rgb({red}, {green}, {blue}); color: {text_color}; padding: 2px;" ' \
                    f'title="Difference: {color_values[i]:.4f}, Original logprob: {original_log_probs[i]:.4f}, Ablated logprob: {ablated_log_probs[i]:.4f}">{visible_string}</span> '
        else:
            # Generate the HTML with background color, text color, and tooltip for each string
            html += f'<span style="background-color: rgb({red}, {green}, {blue}); color: {text_color}; padding: 2px;" ' \
                    f'title="{color_values[i]:.4f}">{visible_string}</span> '
    
    html += "</div>"
    
    # Print the HTML in Jupyter Notebook
    display(HTML(html))

def get_weird_tokens(model: HookedTransformer, w_e_threshold=0.4, w_u_threshold=15, plot_norms=False) -> Int[Tensor, "d_vocab"]:
    w_u_norm = model.W_U.norm(dim=0)
    w_e_norm = model.W_E.norm(dim=1)
    w_u_ignore = torch.argwhere(w_u_norm > w_u_threshold).flatten()
    w_e_ignore = torch.argwhere(w_e_norm < w_e_threshold).flatten()
    all_ignore = torch.argwhere((w_u_norm > w_u_threshold) | (w_e_norm < w_e_threshold)).flatten()
    not_ignore = torch.argwhere((w_u_norm <= w_u_threshold) & (w_e_norm >= w_e_threshold)).flatten()

    if plot_norms:
        print(f"Number of W_U neurons to ignore: {len(w_u_ignore)}")
        print(f"Number of W_E neurons to ignore: {len(w_e_ignore)}")
        print(f"Number of unique W_U and W_E neurons to ignore: {len(all_ignore)}")

        fig = px.line(w_u_norm.cpu().numpy(), title="W_U Norm", labels={"value": "W_U.norm(dim=0)", "index": "Vocab Index"})
        fig.add_hline(y=w_u_threshold, line_dash="dash", line_color="red")
        fig.show()
        fig = px.line(w_e_norm.cpu().numpy(), title="W_E Norm", labels={"value": "W_E.norm(dim=1)", "index": "Vocab Index"})
        fig.add_hline(y=w_e_threshold, line_dash="dash", line_color="red")
        fig.show()
    return all_ignore, not_ignore

def top_k_with_exclude(activations: Tensor, k: int, exclude: Tensor, largest=True) -> Tuple[Tensor, Tensor]:
    """Returns the top k activations and indices excluding the indices in exclude.

    Args:
        activations (Tensor): Activations to find the top k of.
        k (int): Number of top activations to return.
        exclude (Tensor): Indices to exclude.

    Returns:
        Tuple[Tensor, Tensor]: Top k activations and indices.
    """
    if len(activations) < k + len(exclude):
        warnings.warn(f"Length of activations ({len(activations)}) is less than k ({k}) + length of exclude ({len(exclude)}).")
        k = len(activations) - len(exclude)
    #assert len(activations) >= k + len(exclude), f"Length of activations ({len(activations)}) must be greater than k ({k}) + length of exclude ({len(exclude)})."
    activations = activations.clone()
    if largest:
        activations[exclude] = -float("inf")
    else:
        activations[exclude] = float("inf")
    top_k_values, top_k_indices = torch.topk(activations, k=k, largest=largest)
    return top_k_values, top_k_indices


def get_frozen_loss_difference_for_component(
        prompts: List[str],
        model: HookedTransformer,
        ablation_hooks=[],
        freeze_act_names=[],
        crop_context_end: None | int=None,
        disable_progress_bar=False
    ):
    original_losses = []
    ablated_losses = []
    frozen_losses = []
    for prompt in tqdm(prompts, disable=disable_progress_bar):
        if crop_context_end is not None:
            tokens = model.to_tokens(prompt)[:, :crop_context_end]
        else:
            tokens = model.to_tokens(prompt)

        original_loss, ablated_loss, original_cache, ablated_cache = get_caches_single_prompt(
            prompt, model, fwd_hooks=ablation_hooks, crop_context_end=crop_context_end, return_type="loss")

        # ['embed', '0_attn_out', '0_mlp_out', '1_attn_out', '1_mlp_out', '2_attn_out', '2_mlp_out', '3_attn_out', '3_mlp_out', '4_attn_out', '4_mlp_out', '5_attn_out', '5_mlp_out']
        # Ablate at the final residual stream value to remove the direct component output
        def freeze_hook(value, hook: HookPoint):
            value = original_cache[hook.name]
            return value            
        
        freeze_hooks = [(freeze_act_name, freeze_hook) for freeze_act_name in freeze_act_names]

        with model.hooks(fwd_hooks=freeze_hooks+ablation_hooks):
            frozen_loss = model(tokens, return_type="loss")

        original_losses.append(original_loss)
        ablated_losses.append(ablated_loss)
        frozen_losses.append(frozen_loss.item())


    print(f"Original loss: {np.mean(original_losses):.2f}, frozen loss: {np.mean(frozen_losses):.2f} (+{((np.mean(frozen_losses) - np.mean(original_losses)) / np.mean(original_losses))*100:.2f}%), ablated loss: {np.mean(ablated_losses):.2f} (+{((np.mean(ablated_losses) - np.mean(original_losses)) / np.mean(original_losses))*100:.2f}%)")
    return np.mean(original_losses), np.mean(ablated_losses), np.mean(frozen_losses)


def get_neuron_unembed(model, neuron, layer, mean_activation_active, mean_activation_inactive, top_k=20, plot=False):
    neuron_weight = model.W_out[layer, neuron].view(1, 1, -1)
    neuron_direction_active = neuron_weight * mean_activation_active 
    neuron_direction_inactive = neuron_weight * mean_activation_inactive 

    tokens_active = model.unembed(neuron_direction_active)
    tokens_inactive = model.unembed(neuron_direction_inactive)
    # Active: German neuron is active - we expect German tokens boosted
    # Inactive: German neuron is inactive - we expect no boost to German tokens
    # Active - Inactive: If the neuron boosts German tokens, we expect this to be positive
    token_differences = (tokens_active - tokens_inactive).flatten()

    all_ignore, _ = get_weird_tokens(model, plot_norms=False)

    boosted_values, boosted_tokens = top_k_with_exclude(token_differences, top_k, exclude=all_ignore)
    inhibited_values, inhibited_tokens = top_k_with_exclude(token_differences, top_k, largest=False, exclude=all_ignore)
    boosted_labels = model.to_str_tokens(boosted_tokens)
    inhibited_labels = model.to_str_tokens(inhibited_tokens)

    if plot:
        assert top_k < 300, "Too many tokens to plot"
        fig = line(x=boosted_values.cpu().numpy(), xticks=boosted_labels, title=f"Boosted tokens from active L{layer}N{neuron}", width=1200)
        fig = line(x=inhibited_values.cpu().numpy(), xticks=inhibited_labels, title=f"Inhibited tokens from active L{layer}N{neuron}", width=1200)


    return boosted_tokens, inhibited_tokens


def get_frozen_loss_difference_position(
        prompt: str,
        model: HookedTransformer,
        ablation_hooks=[],
        freeze_act_names=[],
        crop_context_end: None | int=None,
        pos=-1
    ):

    if crop_context_end is not None:
        tokens = model.to_tokens(prompt)[:, :crop_context_end]
    else:
        tokens = model.to_tokens(prompt)

    original_loss, original_cache = model.run_with_cache(tokens, return_type="loss", loss_per_token=True)
    with model.hooks(fwd_hooks=ablation_hooks):
        ablated_loss, ablated_cache = model.run_with_cache(tokens, return_type="loss", loss_per_token=True)

    # ['embed', '0_attn_out', '0_mlp_out', '1_attn_out', '1_mlp_out', '2_attn_out', '2_mlp_out', '3_attn_out', '3_mlp_out', '4_attn_out', '4_mlp_out', '5_attn_out', '5_mlp_out']
    # Ablate at the final residual stream value to remove the direct component output
    def freeze_hook(value, hook: HookPoint):
        value = original_cache[hook.name]
        return value            
    
    freeze_hooks = [(freeze_act_name, freeze_hook) for freeze_act_name in freeze_act_names]

    with model.hooks(fwd_hooks=freeze_hooks+ablation_hooks):
        frozen_loss = model(tokens, return_type="loss", loss_per_token=True)

    original_loss = original_loss[0, pos].item()
    ablated_loss = ablated_loss[0, pos].item()
    frozen_loss = frozen_loss[0,pos].item()

    print(f"Loss on token {model.to_str_tokens(tokens[0, pos-1])} -> {model.to_str_tokens(tokens[0, pos])}")
    print(f"Original loss: {original_loss:.2f}, direct loss of context neuron: {frozen_loss:.2f} (+{((frozen_loss - original_loss) / original_loss)*100:.2f}%), ablated loss: {ablated_loss:.2f} (+{((ablated_loss - original_loss) / original_loss)*100:.2f}%)")
    return original_loss, ablated_loss, frozen_loss


def get_ablated_loss_difference_position(
        prompt: str,
        model: HookedTransformer,
        ablation_hooks=[],
        ablate_act_names=[],
        crop_context_end: None | int=None,
        disable_progress_bar=False,
        pos=-1
    ):
    original_losses = []
    ablated_losses = []
    frozen_losses = []

    if crop_context_end is not None:
        tokens = model.to_tokens(prompt)[:, :crop_context_end]
    else:
        tokens = model.to_tokens(prompt)

    original_loss, original_cache = model.run_with_cache(tokens, return_type="loss", loss_per_token=True)
    with model.hooks(fwd_hooks=ablation_hooks):
        ablated_loss, ablated_cache = model.run_with_cache(tokens, return_type="loss", loss_per_token=True)

    # ['embed', '0_attn_out', '0_mlp_out', '1_attn_out', '1_mlp_out', '2_attn_out', '2_mlp_out', '3_attn_out', '3_mlp_out', '4_attn_out', '4_mlp_out', '5_attn_out', '5_mlp_out']
    # Ablate at the final residual stream value to remove the direct component output
    def freeze_hook(value, hook: HookPoint):
        value = ablated_cache[hook.name]
        return value            
    
    freeze_hooks = [(freeze_act_name, freeze_hook) for freeze_act_name in ablate_act_names]

    with model.hooks(fwd_hooks=freeze_hooks):
        frozen_loss = model(tokens, return_type="loss", loss_per_token=True)

    original_losses.append(original_loss[0, pos].item())
    ablated_losses.append(ablated_loss[0, pos].item())
    frozen_losses.append(frozen_loss[0,pos].item())

    print(f"Loss on token {model.to_str_tokens(tokens[0, pos-1])} -> {model.to_str_tokens(tokens[0, pos])}")
    print(f"Original loss: {np.mean(original_losses):.2f}, indirect loss of later components: {np.mean(frozen_losses):.2f} (+{((np.mean(frozen_losses) - np.mean(original_losses)) / np.mean(original_losses))*100:.2f}%), ablated loss: {np.mean(ablated_losses):.2f} (+{((np.mean(ablated_losses) - np.mean(original_losses)) / np.mean(original_losses))*100:.2f}%)")
    return np.mean(original_losses), np.mean(ablated_losses), np.mean(frozen_losses)

def get_frozen_logits(prompt: str, model: HookedTransformer, ablation_hooks, freeze_act_names, crop_context_end=None):
    
    if crop_context_end is not None:
        tokens = model.to_tokens(prompt)[:, :crop_context_end]
    else:
        tokens = model.to_tokens(prompt)

    original_loss, original_cache = model.run_with_cache(tokens, return_type="loss", loss_per_token=True)
    

    # ['embed', '0_attn_out', '0_mlp_out', '1_attn_out', '1_mlp_out', '2_attn_out', '2_mlp_out', '3_attn_out', '3_mlp_out', '4_attn_out', '4_mlp_out', '5_attn_out', '5_mlp_out']
    # Ablate at the final residual stream value to remove the direct component output
    def freeze_hook(value, hook: HookPoint):
        value = original_cache[hook.name]
        return value            
    
    freeze_hooks = [(freeze_act_name, freeze_hook) for freeze_act_name in freeze_act_names]

    with model.hooks(fwd_hooks=freeze_hooks+ablation_hooks):
        frozen_logits = model(tokens, return_type="logits", loss_per_token=True)

    return frozen_logits


def get_ablated_logits(prompt: str, model: HookedTransformer, ablation_hooks, freeze_act_names, crop_context_end=None):
    
    if crop_context_end is not None:
        tokens = model.to_tokens(prompt)[:, :crop_context_end]
    else:
        tokens = model.to_tokens(prompt)
    
    with model.hooks(fwd_hooks=ablation_hooks):
        ablated_loss, ablated_cache = model.run_with_cache(tokens, return_type="loss", loss_per_token=True)

    # ['embed', '0_attn_out', '0_mlp_out', '1_attn_out', '1_mlp_out', '2_attn_out', '2_mlp_out', '3_attn_out', '3_mlp_out', '4_attn_out', '4_mlp_out', '5_attn_out', '5_mlp_out']
    # Ablate at the final residual stream value to remove the direct component output
    def freeze_hook(value, hook: HookPoint):
        value = ablated_cache[hook.name]
        return value            
    
    freeze_hooks = [(freeze_act_name, freeze_hook) for freeze_act_name in freeze_act_names]

    with model.hooks(fwd_hooks=freeze_hooks):
        frozen_logits = model(tokens, return_type="logits", loss_per_token=True)

    return frozen_logits

def get_frozen_loss_difference_measure(
        prompt: str,
        model: HookedTransformer,
        ablation_hooks=[],
        freeze_act_names=("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out", "blocks.5.hook_mlp_out"),
        crop_context_end: None | int=None,
        debug_log=True
    ):
    """ 
    Ablates context neuron via ablation_hooks but freezes some later components listed in freeze_act_names with the clean unablated activations. 
    Return the difference of the full ablated loss and the loss with some frozen components. 
    This allows to remove the effect of non frozen components from the total loss. 
    By default, all later components are frozen, resulting in the direct effect of the context neuron being removed from the overall loss. 

    Args:
        prompt (str): _description_
        model (HookedTransformer): _description_
        ablation_hooks (list, optional): _description_. Defaults to [].
        freeze_act_names (tuple, optional): _description_. Defaults to ("blocks.4.hook_attn_out", "blocks.5.hookattn_out", "blocks.4.hook_mlp_out", "blocks.5.hook_mlp_out").
        crop_context_end (None | int, optional): _description_. Defaults to None._

    Returns:
        _type_: _description_
    """

    if crop_context_end is not None:
        tokens = model.to_tokens(prompt)[:, :crop_context_end]
    else:
        tokens = model.to_tokens(prompt)

    original_loss, original_cache = model.run_with_cache(tokens, return_type="loss", loss_per_token=True)
    with model.hooks(fwd_hooks=ablation_hooks):
        ablated_loss, ablated_cache = model.run_with_cache(tokens, return_type="loss", loss_per_token=True)

    # ['embed', '0_attn_out', '0_mlp_out', '1_attn_out', '1_mlp_out', '2_attn_out', '2_mlp_out', '3_attn_out', '3_mlp_out', '4_attn_out', '4_mlp_out', '5_attn_out', '5_mlp_out']
    # Ablate at the final residual stream value to remove the direct component output
    def freeze_hook(value, hook: HookPoint):
        value = original_cache[hook.name]
        return value            
    
    freeze_hooks = [(freeze_act_name, freeze_hook) for freeze_act_name in freeze_act_names]

    with model.hooks(fwd_hooks=freeze_hooks+ablation_hooks):
        frozen_loss = model(tokens, return_type="loss", loss_per_token=True)

    # Frozen loss is the direct loss effect of context neuron
    # Ablated loss is the total loss effect of context neuron + later components
    # We care about high loss from later components 
    loss_increase = ablated_loss[0,:] - frozen_loss[0,:]

    if debug_log:
        # print("No effect (original loss)", original_loss[0,:5])
        print("Loss from ablated context neuron (total effect)", ablated_loss[0,:])
        print("Loss from ablating context neuron but restoring later activations (direct effect)", frozen_loss[0,:])
        print("Indirect effect: total_effect - direct effect", loss_increase)

    #print(f"Original loss: {np.max(original_loss):.2f}, frozen loss: {np.max(frozen_loss):.2f} (+{((np.mean(frozen_loss) - np.mean(original_loss)) / np.mean(original_losses))*100:.2f}%), ablated loss: {np.mean(ablated_losses):.2f} (+{((np.mean(ablated_losses) - np.mean(original_losses)) / np.mean(original_losses))*100:.2f}%)")
    return loss_increase

def get_ablated_loss_difference_measure(
        prompt: str,
        model: HookedTransformer,
        ablation_hooks=[],
        freeze_act_names=("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out", "blocks.5.hook_mlp_out"),
        crop_context_end: None | int=None,
        debug_log=True
    ):
    """ Runs model without ablating the context neuron but inserts corrupted caches from ablated context neuron into some later components. Returns the difference between the fully 
    corrupted and the partially corrupted run. By default, all later components are corrupted, resulting in the indirect effect of ablating the context neuron.
    """

    if crop_context_end is not None:
        tokens = model.to_tokens(prompt)[:, :crop_context_end]
    else:
        tokens = model.to_tokens(prompt)

    with model.hooks(fwd_hooks=ablation_hooks):
        ablated_loss, ablated_cache = model.run_with_cache(tokens, return_type="loss", loss_per_token=True)

    # ['embed', '0_attn_out', '0_mlp_out', '1_attn_out', '1_mlp_out', '2_attn_out', '2_mlp_out', '3_attn_out', '3_mlp_out', '4_attn_out', '4_mlp_out', '5_attn_out', '5_mlp_out']
    # Ablate at the final residual stream value to remove the direct component output
    def freeze_hook(value, hook: HookPoint):
        value = ablated_cache[hook.name]
        return value            
    
    freeze_hooks = [(freeze_act_name, freeze_hook) for freeze_act_name in freeze_act_names]

    with model.hooks(fwd_hooks=freeze_hooks):
        frozen_loss = model(tokens, return_type="loss", loss_per_token=True)


    # Frozen loss is the direct loss effect of context neuron
    # Ablated loss is the total loss effect of context neuron + later components
    # We care about high loss from later components 
    loss_increase = ablated_loss[0,:] - frozen_loss[0,:]

    if debug_log:
        print("Total effect (loss from ablated context neuron)", ablated_loss[0,:])
        print("Indirect effect (loss from not ablating context neuron but corrupting later activations)", frozen_loss[0,:])
        print("Direct effect (total_effect - indirect effect)", loss_increase)

    #print(f"Original loss: {np.max(original_loss):.2f}, frozen loss: {np.max(frozen_loss):.2f} (+{((np.mean(frozen_loss) - np.mean(original_loss)) / np.mean(original_losses))*100:.2f}%), ablated loss: {np.mean(ablated_losses):.2f} (+{((np.mean(ablated_losses) - np.mean(original_losses)) / np.mean(original_losses))*100:.2f}%)")
    return loss_increase


def split_effects(
        prompt: str,
        model: HookedTransformer,
        ablation_hooks=[],
        freeze_act_names=("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out", "blocks.5.hook_mlp_out"),
        debug_log=True,
        return_absolute=False
    ) -> Tuple[Tensor, Tensor, Tensor]:
    """Gets the absolute contribution to loss of an ablation's direct effect, indirect effect, and total effect. Also returns the component of loss unaffected by the ablation."""
    original_loss, original_cache = model.run_with_cache(prompt, return_type="loss", loss_per_token=True)
    with model.hooks(fwd_hooks=ablation_hooks):
        ablated_loss, ablated_cache = model.run_with_cache(prompt, return_type="loss", loss_per_token=True)

    # Add the effects of ablating at MLP3 to the components after MLP3
    def freeze_ablated_hook(value, hook: HookPoint):
        value = ablated_cache[hook.name]
        return value             
    freeze_ablated_hooks = [(freeze_act_name, freeze_ablated_hook) for freeze_act_name in freeze_act_names]
    with model.hooks(fwd_hooks=freeze_ablated_hooks):
        original_with_frozen_ablated = model(prompt, return_type="loss", loss_per_token=True)

    # Remove the effects of ablating at MLP3 from the components after MLP3
    def freeze_original_hook(value, hook: HookPoint):
        value = original_cache[hook.name]
        return value         
    freeze_original_hooks = [(freeze_act_name, freeze_original_hook) for freeze_act_name in freeze_act_names]
    with model.hooks(fwd_hooks=freeze_original_hooks+ablation_hooks):
        ablated_with_original_frozen_loss = model(prompt, return_type="loss", loss_per_token=True)

    # Total effect of ablating German context neuron is total change in loss
    total_effect_loss_change = ablated_loss - original_loss
    # Direct effect of ablating German context neuron is change in loss from ablating German context neuron but restoring later activations
    direct_effect_loss_change = ablated_with_original_frozen_loss - original_loss
    # Indirect effect of ablating German context neuron is change in loss from patching in ablated later activations
    indirect_effect_loss_change = original_with_frozen_ablated - original_loss

    if debug_log:
        print("original loss", original_loss[0,:5])
        print("total effect change in loss", total_effect_loss_change[0,:5])
        print("direct effect change in loss", direct_effect_loss_change[0,:5])
        print("indirect effect change in loss", indirect_effect_loss_change[0,:5])

    if return_absolute:
        return original_loss, ablated_loss, ablated_with_original_frozen_loss, original_with_frozen_ablated
    else:
        print("Warning: returning relative loss changes.")
        #print(f"Original loss: {np.max(original_loss):.2f}, frozen loss: {np.max(frozen_loss):.2f} (+{((np.mean(frozen_loss) - np.mean(original_loss)) / np.mean(original_losses))*100:.2f}%), ablated loss: {np.mean(ablated_losses):.2f} (+{((np.mean(ablated_losses) - np.mean(original_losses)) / np.mean(original_losses))*100:.2f}%)")
        return original_loss, total_effect_loss_change, direct_effect_loss_change, indirect_effect_loss_change
    

def plot_barplot(data: list[list[float]], names: list[str], short_names = None, xlabel="", ylabel="", title="", width=1000, show=True, legend=True):
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)

    fig = go.Figure()
    if short_names is None:
        short_names = names
    for i in range(len(names)):
        fig.add_trace(go.Bar(
            x=[short_names[i]],
            y=[means[i]],
            error_y=dict(
                type='data',
                array=[stds[i]],
                visible=True
            ),
            name=names[i]
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        barmode='group',
        width=width,
        showlegend=legend
    )
    
    if show:
        fig.show()
    else: 
        return fig


# New

def get_mlp5_attribution_without_mlp4(prompt: str, model: HookedTransformer, ablation_hooks: list, pos: int | None = -1):
    # Freeze everything except for MLP5 to see if MLP5 depends on MLP4
    freeze_act_names=("blocks.5.hook_attn_out", "blocks.4.hook_attn_out", "blocks.4.hook_mlp_out")
    original_loss, total_effect_loss, direct_mlp3_mlp5_loss, _= split_effects(prompt, model, ablation_hooks=ablation_hooks, freeze_act_names=freeze_act_names, debug_log=False, return_absolute=True)
    freeze_act_names=("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out", "blocks.5.hook_mlp_out")
    _, _, direct_mlp3_loss, _ = split_effects(prompt, model, ablation_hooks=ablation_hooks, freeze_act_names=freeze_act_names, debug_log=False, return_absolute=True)

    if pos is not None:
        return original_loss[0, pos].item(), total_effect_loss[0, pos].item(), direct_mlp3_mlp5_loss[0, pos].item(), direct_mlp3_loss[0, pos].item()
    else:
        return original_loss[0, :], total_effect_loss[0, :], direct_mlp3_mlp5_loss[0, :], direct_mlp3_loss[0, :]


def get_mlp3_4_attribution(prompt: str, model: HookedTransformer, ablation_hooks: list, pos: int | None = -1):
    # Freeze everything except for MLP5 to see if MLP5 depends on MLP4
    freeze_act_names=("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.5.hook_mlp_out")
    original_loss, total_effect_loss, direct_mlp3_mlp4_loss, _= split_effects(prompt, model, ablation_hooks=ablation_hooks, freeze_act_names=freeze_act_names, debug_log=False, return_absolute=True)

    if pos is not None:
        return direct_mlp3_mlp4_loss[0, pos].item()
    else:
        return direct_mlp3_mlp4_loss[0, :]


def get_neuron_logit_contribution(cache: ActivationCache, model: HookedTransformer, answer_tokens: Int[Tensor, "batch pos"], layer: int, pos:int | None) -> Float[Tensor, "neuron pos"]:
    # Expects cache from a single example, won't work on batched examples
    # Get per neuron output of MLP layer
    neuron_directions = cache.get_neuron_results(layer, neuron_slice=utils.Slice(input_slice=None), pos_slice=utils.Slice(input_slice=None))
    neuron_directions = einops.rearrange(neuron_directions, 'batch pos neuron residual -> neuron batch pos residual')
    # We need to apply the final layer norm because the unembed operation is applied after the final layer norm, so the answer token
    # directions are in the same space as the final layer norm output
    # LN leads to finding top tokens with slightly higher loss attribution
    if pos is None:
        scaled_neuron_directions = cache.apply_ln_to_stack(neuron_directions)[:, 0, :-1, :]
    else:
        scaled_neuron_directions = cache.apply_ln_to_stack(neuron_directions)[:, 0, pos, :] # [neuron embed]
    # Unembed of correct answer tokens
    correct_token_directions = model.W_U[:, answer_tokens].squeeze(1) # [embed] 
    # Neuron attribution to correct answer token by position
    if pos is None:
        unembedded = einops.einsum(scaled_neuron_directions, correct_token_directions, 'neuron pos residual, residual pos -> pos neuron')
    else:
        unembedded = einops.einsum(scaled_neuron_directions, correct_token_directions, 'neuron residual, residual -> neuron')
    return unembedded

def MLP_attribution(prompt: str, model: HookedTransformer, fwd_hooks, layer_to_compare=5, pos: int | None =-1):
    
    tokens = model.to_tokens(prompt)
    if pos is None:
        answer_tokens = tokens[:, 1:]
    else:
        answer_tokens = tokens[:, pos+1] #TODO check
    # Get difference between ablated and unablated neurons' contribution to answer logit
    _, _, original_cache, ablated_cache = get_caches_single_prompt(
        prompt, model, fwd_hooks)
    original_unembedded = get_neuron_logit_contribution(original_cache, model, answer_tokens, layer=layer_to_compare, pos=pos) # [neuron]
    ablated_unembedded = get_neuron_logit_contribution(ablated_cache, model, answer_tokens, layer=layer_to_compare, pos=pos)
    differences = (original_unembedded - ablated_unembedded).detach().cpu() # [neuron]
    return differences

def get_neuron_loss_attribution(prompt, model, neurons, ablation_hooks: list, pos: int | None=-1, layer=5):
    original_loss, original_cache = model.run_with_cache(prompt, return_type="loss", loss_per_token=True)
    with model.hooks(fwd_hooks=ablation_hooks):
        ablated_loss, ablated_cache = model.run_with_cache(prompt, return_type="loss", loss_per_token=True)

    # Remove the effects of ablating at MLP3 from the components after MLP3
    def freeze_neurons_hook(value, hook: HookPoint):
        if pos is not None:
            value[:, :, neurons] = original_cache[hook.name][:, :, neurons] # [batch pos neuron]
        else:
            src_tmp = original_cache[hook.name][0, :-1, :].gather(dim=-1, index=neurons.cuda())
            value[0, :-1, :] = value[0, :-1, :].scatter_(dim=-1, index=neurons.cuda(), src=src_tmp)
        return value      

    freeze_original_hooks = [(f"blocks.{layer}.mlp.hook_post", freeze_neurons_hook)]
    with model.hooks(fwd_hooks=ablation_hooks+freeze_original_hooks):
        ablated_with_original_frozen_loss = model(prompt, return_type="loss", loss_per_token=True)
    #print(ablated_loss[0, :], ablated_with_original_frozen_loss[0, :])
    if pos is not None:
        return original_loss[0, pos].item(), ablated_loss[0, pos].item(), ablated_with_original_frozen_loss[0, pos].item()
    else:
        return original_loss[0, :], ablated_loss[0, :], ablated_with_original_frozen_loss[0, :]
    
def pos_wise_mlp_effect_on_single_prompt(prompt: str, model:HookedTransformer, ablation_hooks: list, k = 20, log=False, top_neurons=None, answer_pos=None, return_mlp4_less_mlp5=False):
    
    if answer_pos is not None:
        pos = answer_pos
        assert pos != 0, "First answer position = 1"
        if pos < 0:
            pos = model.to_tokens(prompt).shape[1]-1+pos
        elif pos > 0:
            pos -= 1
    else:
        pos = None

    if (top_neurons is not None) and (len(top_neurons) < k):
            print(f"Warning: Only {len(top_neurons)} neurons given for k={k}.")

    original_loss, total_effect_loss, direct_mlp3_mlp5_loss, direct_mlp3_loss = get_mlp5_attribution_without_mlp4(prompt, model, ablation_hooks=ablation_hooks, pos=pos)
    direct_mlp3_mlp4_loss = get_mlp3_4_attribution(prompt, model, ablation_hooks=ablation_hooks, pos=pos)
    if top_neurons is None:
        differences = MLP_attribution(prompt, model, fwd_hooks=ablation_hooks, layer_to_compare=5, pos=pos)
        # Shape (pos, k)
        top_diff, top_diff_neurons = torch.topk(differences, k, largest=True)
        # print(top_diff_neurons)
    else:
        top_diff_neurons = torch.LongTensor(top_neurons)
    
    # also ablates attention 4
    if return_mlp4_less_mlp5:
        with model.hooks(fwd_hooks=ablation_hooks):
            _, ablated_cache = model.run_with_cache(prompt)
        def deactivate_component_hook(value, hook: HookPoint):
            value = ablated_cache[hook.name]
            return value
        deactivate_4_hooks = [(f"blocks.4.mlp.hook_pre", deactivate_component_hook)]
        # deactivate_4_hooks = [(f"blocks.4.hook_attn_out", deactivate_component_hook), (f"blocks.4.mlp.hook_pre", deactivate_component_hook)]
        with model.hooks(fwd_hooks=deactivate_4_hooks):
            # TODO Frozen loss does not match for single position and pos=None
            if pos is None:
                _, _, frozen_loss_inactive_mlp4 = get_neuron_loss_attribution(prompt, model, top_diff_neurons[:, :k], ablation_hooks=ablation_hooks, pos=pos)
            else:
                _, _, frozen_loss_inactive_mlp4 = get_neuron_loss_attribution(prompt, model, top_diff_neurons[:k], ablation_hooks=ablation_hooks, pos=pos)
    
    if pos is None:
            _, _, frozen_loss = get_neuron_loss_attribution(prompt, model, top_diff_neurons[:, :k], ablation_hooks=ablation_hooks, pos=pos)
    else:
        _, _, frozen_loss = get_neuron_loss_attribution(prompt, model, top_diff_neurons[:k], ablation_hooks=ablation_hooks, pos=pos)

    ablation_loss_increase = total_effect_loss - original_loss
    frozen_loss_decrease = total_effect_loss - frozen_loss

    if log and (pos is not None):
        print(f"\n{prompt}")
        print(f"Original loss: {original_loss:.4f}")
        print(f"Total effect loss: {total_effect_loss:.4f}")#
        print(f"Direct effect loss of MLP3 and MLP5 (restoring MLP4 and attention): {direct_mlp3_mlp5_loss:.4f}")
        print(f"Direct effect loss of MLP3 (restoring MLP4 and MLP5 and attention): {direct_mlp3_loss:.4f}")
        print(f"Total effect loss when freezing top MLP5 neurons: {frozen_loss:.4f}")
    elif log:
        print(f"\n{prompt}")
        print(f"Original loss: {original_loss}")
        print(f"Total effect loss: {total_effect_loss}")#
        print(f"Direct effect loss of MLP3 and MLP5 (restoring MLP4 and attention): {direct_mlp3_mlp5_loss}")
        print(f"Direct effect loss of MLP3 (restoring MLP4 and MLP5 and attention): {direct_mlp3_loss}")
        print(f"Total effect loss when freezing top MLP5 neurons: {frozen_loss}")
    
    if return_mlp4_less_mlp5:
        return original_loss, total_effect_loss, direct_mlp3_mlp5_loss, direct_mlp3_loss, frozen_loss, frozen_loss_inactive_mlp4
    else:
        return original_loss, total_effect_loss, direct_mlp3_mlp5_loss, direct_mlp3_loss, frozen_loss



def MLP_attribution_4_5(prompt: str, model: HookedTransformer, fwd_hooks, layer_to_compare=5, pos: int | None =-1):
    # neurons with top differences when ablated 3 is patched in vs ablated 3 AND ablated 4
    tokens = model.to_tokens(prompt)
    if pos is None:
        answer_tokens = tokens[:, 1:]
    else:
        answer_tokens = tokens[:, pos+1] #TODO check

    # Get difference between ablated and unablated neurons' contribution to answer logit
    _, _, original_cache, ablated_cache = get_caches_single_prompt(
        prompt, model, fwd_hooks)
    
    with model.hooks(fwd_hooks=fwd_hooks+[(f"blocks.4.mlp.hook_pre", lambda value, hook: ablated_cache[hook.name])]):
        logits, mlp_3_4_disabled_cache = model.run_with_cache(prompt)

    mlp_3_4_disabled_unembedded = get_neuron_logit_contribution(mlp_3_4_disabled_cache, model, answer_tokens, layer=layer_to_compare, pos=pos) # [neuron]
    mlp_3_disabled_unembedded = get_neuron_logit_contribution(ablated_cache, model, answer_tokens, layer=layer_to_compare, pos=pos)
    differences = (mlp_3_disabled_unembedded - mlp_3_4_disabled_unembedded).detach().cpu() # [neuron]
    return differences


def effect_of_context_and_mlp_4_on_neurons_vs_just_context(prompt: str, model:HookedTransformer, ablation_hooks: list, k = 20, log=False, top_neurons=None, answer_pos=-1, return_mlp4_less_mlp5=False):
    # We have a subset of neurons which are different when the context neuron is disabled. We want a group of neurons which are different when context neuron is disabled, vs. context 
    # neuron Identify neurons which 
    pos = model.to_tokens(prompt).shape[1]-1+answer_pos

    original_loss, total_effect_loss, direct_mlp3_mlp5_loss, direct_mlp3_loss = get_mlp5_attribution_without_mlp4(prompt, model, ablation_hooks=ablation_hooks, pos=pos)

    if top_neurons is None:
        differences = MLP_attribution(prompt, model, fwd_hooks=ablation_hooks, layer_to_compare=5, pos=pos)
        differences_with_4 = MLP_attribution(prompt, model, fwd_hooks=ablation_hooks, layer_to_compare=5, pos=pos)
        top_diff, top_diff_neurons = torch.topk(differences, k, largest=True) # Shape (pos, k)
    else:
        top_diff_neurons = torch.LongTensor(top_neurons)
    
    differences_with_34_disabled = MLP_attribution_4_5(prompt, model, fwd_hooks=ablation_hooks, layer_to_compare=5, pos=pos)    
    top_diff_3_4_disabled, top_diff_neurons_3_4_disabled = torch.topk(differences_with_34_disabled, 200, largest=True)
    # print(top_diff_neurons_3_4_disabled)
    # indices = torch.zeros_like(top_diff_neurons, dtype = torch.uint8).cpu()
    # for elem in top_diff_neurons_3_4_disabled:
    #     indices = indices | (top_diff_neurons == elem.cpu())  
    # intersection = top_diff_neurons[indices]  
    # print(intersection)

    if return_mlp4_less_mlp5:
        with model.hooks(fwd_hooks=ablation_hooks):
            _, ablated_cache = model.run_with_cache(prompt)
        def deactivate_component_hook(value, hook: HookPoint):
            value = ablated_cache[hook.name]
            return value
        deactivate_4_hooks = [(f"blocks.4.mlp.hook_pre", deactivate_component_hook)]

        # Currently we look at neurons that contribute differently to the answer token when MLP3 is ablated vs. not. 
        # Within this subset, some neurons read directly from MLP3. Some are more complex and also read from MLP4 (3 + 4, not 3 via 4).

        # For these, they won't activate as differently when enabling MLP3 but patching in a disabled MLP4, vs. on the original cache.
        # They will be a subset of top_neurons. We want to list out the top_neurons but how different their activation is in this scenario.
        
        # Task: Disable MLP4 and see how the MLP5 neurons contribute differently to the answer token.

        # patching in the clean MLP4 on its own doesn't help with loss. But context neuron + MLP4 helps some of the top MLP5 neurons be more different.
        # We can identify these by looking at which of the top neurons are more different when we patch in the clean MLP4 - but which direction?

        # we want to split the top neurons into ones which read from 4 and ones which read from 3. Basically, if we ablate the context neuron but patch in the clean 4 some of these
        # neurons will be very different. These are our AND neurons.
        with model.hooks(fwd_hooks=deactivate_4_hooks):
            _, _, frozen_loss_inactive_mlp4 = get_neuron_loss_attribution(prompt, model, top_diff_neurons[:k], ablation_hooks=ablation_hooks, pos=pos)
        _, _, frozen_loss = get_neuron_loss_attribution(prompt, model, top_diff_neurons[:k], ablation_hooks=ablation_hooks, pos=pos)

    ablation_loss_increase = total_effect_loss - original_loss
    frozen_loss_decrease = total_effect_loss - frozen_loss
    
    if return_mlp4_less_mlp5:
        return original_loss, total_effect_loss, direct_mlp3_mlp5_loss, direct_mlp3_loss, frozen_loss, frozen_loss_inactive_mlp4, top_diff_neurons, top_diff_neurons_3_4_disabled
    else:
        return original_loss, total_effect_loss, direct_mlp3_mlp5_loss, direct_mlp3_loss, frozen_loss, frozen_loss_inactive_mlp4, top_diff_neurons, top_diff_neurons_3_4_disabled


def get_direct_effect(prompt: str | list[str], model: HookedTransformer, context_ablation_hooks: list, context_activation_hooks: list, pos: int | None = -1,
                      deactivated_components=("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out"),
                      activated_components=("blocks.5.hook_mlp_out",), return_type: Literal['logits', 'logprobs', 'loss'] = 'loss'
):
    """ Direct MLP5 effect

    Args:
        prompt (str): _description_
        model (HookedTransformer): _description_
        context_ablation_hooks (list): _description_
        context_activation_hooks (list): _description_
        pos (int | None, optional): _description_. Defaults to -1.
        deactivated_components (tuple, optional): _description_. Defaults to ("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out").
        activated_components (tuple, optional): _description_. Defaults to ("blocks.5.hook_mlp_out",).

    Returns:
        _type_: _description_
    """
    metric_return_type = 'loss' if return_type == 'loss' else 'logits'
    
    # 1. Cache activations and ablated activations
    with model.hooks(fwd_hooks=context_activation_hooks):
        original_metric = model(prompt, return_type=metric_return_type, loss_per_token=True)
    with model.hooks(fwd_hooks=context_ablation_hooks):
        ablated_metric, ablated_cache = model.run_with_cache(prompt, return_type=metric_return_type, loss_per_token=True)

    # 2. Activate context neuron, ablate deactivated_components, cache activations    
    def deactivate_components_hook(value, hook: HookPoint):
        value = ablated_cache[hook.name]
        return value             
    deactivate_components_hooks = [(freeze_act_name, deactivate_components_hook) for freeze_act_name in deactivated_components]
    with model.hooks(fwd_hooks=deactivate_components_hooks+context_activation_hooks):
        context_and_activated_metric, context_and_activated_cache = model.run_with_cache(prompt, return_type=metric_return_type, loss_per_token=True)

    # 3. Deactivate context neuron, patch context neuron into activated_components, deactivate deactivated_components (doesn't matter when looking at MLP5)
    def activate_components_hook(value, hook: HookPoint):
        value = context_and_activated_cache[hook.name]
        return value         
    activate_components_hooks = [(freeze_act_name, activate_components_hook) for freeze_act_name in activated_components]
    with model.hooks(fwd_hooks=activate_components_hooks+context_ablation_hooks+deactivate_components_hooks):
        only_activated_metric = model(prompt, return_type=metric_return_type, loss_per_token=True)

    # convert logits metric to logprobs metric
    # buggy - prefer converting from logits outside the method
    # if return_type == 'logprobs':
    #     original_metric = original_metric.log_softmax(-1)
    #     ablated_metric = ablated_metric.log_softmax(-1)
    #     context_and_activated_metric = context_and_activated_metric.log_softmax(-1)
    #     only_activated_metric = only_activated_metric.log_softmax(-1)

    # fix bug while maintaining backwards compatibility
    if (return_type == 'logprobs' or return_type == 'logits') and pos is not None:
        return original_metric[:, pos], ablated_metric[:, pos], context_and_activated_metric[:, pos], only_activated_metric[:, pos]

    if original_metric.shape[0] > 1:
        if pos is not None:
            return original_metric[:, pos], ablated_metric[:, pos], context_and_activated_metric[:, pos], only_activated_metric[:, pos],
        else:
            return original_metric, ablated_metric, context_and_activated_metric, only_activated_metric
    if pos is not None:
        return original_metric[0, pos].item(), ablated_metric[0, pos].item(), context_and_activated_metric[0, pos].item(), only_activated_metric[0, pos].item()
    else:
        return original_metric[0, :], ablated_metric[0, :], context_and_activated_metric[0, :], only_activated_metric[0, :]


def relevant_names_filter(name: str):
    return any([name.endswith(s) for s in ["attn_out", "mlp_out"]])


def get_direct_indirect_mlp_effect(prompt: str | list[str], model: HookedTransformer, context_ablation_hooks: list, context_activation_hooks: list, pos: int | None = -1,
                      senders=("blocks.4.hook_mlp_out"), receivers=("blocks.5.hook_mlp_out")
):
    """ Many effect of senders via receivers with attn frozen

    Args:
        prompt (str): _description_
        model (HookedTransformer): _description_
        context_ablation_hooks (list): _description_
        context_activation_hooks (list): _description_
        pos (int | None, optional): _description_. Defaults to -1.
        deactivated_components (tuple, optional): _description_. Defaults to ("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out").
        activated_components (tuple, optional): _description_. Defaults to ("blocks.5.hook_mlp_out",).

    Returns:
        _type_: _description_
    """

    # 1. Activate context neuron, cache activations
    # 2. Deactivate context neuron, cache activations
    # 3. Deactivate context neuron, patch enabled sender, freeze attn, cache activations
    #   Total effect of sender
    # 4. Deactivate context neuron, patch receiver from previous run
    # 	Indirect effect of sender
    # 5. Deactivate context neuron, patch enabled sender, freeze attn, patch disabled receiver
    # 	Direct effect of sender

    # 1. Activate context neuron, cache activations
    with model.hooks(fwd_hooks=context_activation_hooks):
        original_loss, original_cache = model(prompt, return_type="loss", loss_per_token=True)

    # 2. Deactivate context neuron, cache activations
    with model.hooks(fwd_hooks=context_ablation_hooks):
        ablated_loss, ablated_cache = model.run_with_cache(prompt, return_type="loss", loss_per_token=True)

    # 3. Deactivate context neuron, freeze attn, patch enabled sender, cache activations
    def deactivate_attn_hook(value, hook: HookPoint):
        value = ablated_cache[hook.name]
        return value
    deactivate_attn_hooks = [(lambda name: name.endswith("attn_out"), deactivate_attn_hook)]
    def activate_components_hook(value, hook: HookPoint):
        value = original_cache[hook.name]
        return value         
    activate_senders_hooks = [(freeze_act_name, activate_components_hook) for freeze_act_name in senders]
    with model.hooks(fwd_hooks=context_ablation_hooks+deactivate_attn_hooks+activate_senders_hooks):
        total_mlp_effect_loss, senders_activated_cache = model(prompt, return_type="loss", loss_per_token=True)

    # 4. Deactivate context neuron, patch receiver from previous run
    def activate_receivers_hook(value, hook: HookPoint):
        value = senders_activated_cache[hook.name]
        return value  
    activate_receivers_hooks = [(freeze_act_name, activate_receivers_hook) for freeze_act_name in receivers]
    with model.hooks(fwd_hooks=context_ablation_hooks+activate_receivers_hooks):
        indirect_effect_loss, receivers_activated_cache = model.run_with_cache(prompt, return_type="loss", loss_per_token=True)

    # 5. Deactivate context neuron, freeze attn, patch enabled sender, patch disabled receiver
    def deactivate_components_hook(value, hook: HookPoint):
        value = ablated_cache[hook.name]
        return value             
    deactivate_receivers_hooks = [(freeze_act_name, deactivate_components_hook) for freeze_act_name in receivers]
    with model.hooks(fwd_hooks=context_ablation_hooks+deactivate_attn_hooks+activate_senders_hooks+deactivate_receivers_hooks):
        direct_effect_loss, senders_direct_effect_cache = model.run_with_cache(prompt, return_type="loss", loss_per_token=True)


    if original_loss.shape[0] > 1:
        if pos is not None:
            return original_loss[:, pos], ablated_loss[:, pos], indirect_effect_loss[:, pos], total_mlp_effect_loss[:, pos], direct_effect_loss[:, pos]
        else:
            return original_loss, ablated_loss, indirect_effect_loss, total_mlp_effect_loss, direct_effect_loss
    if pos is not None:
        return original_loss[0, pos].item(), ablated_loss[0, pos].item(), indirect_effect_loss[0, pos].item(), total_mlp_effect_loss[0, pos].item(), direct_effect_loss[0, pos].item()
    else:
        return original_loss[0, :], ablated_loss[0, :], indirect_effect_loss[0, :], total_mlp_effect_loss[0, :], direct_effect_loss[0, :]
    

def get_patched_cache(prompt: str | list[str], model: HookedTransformer, context_ablation_hooks: list, context_activation_hooks: list, pos: int | None = -1,
                      deactivated_components=("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out"),
                      activated_components=("blocks.5.hook_mlp_out",)
):
    """ Get cache

    Args:
        prompt (str): _description_
        model (HookedTransformer): _description_
        context_ablation_hooks (list): _description_
        context_activation_hooks (list): _description_
        pos (int | None, optional): _description_. Defaults to -1.
        deactivated_components (tuple, optional): _description_. Defaults to ("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out").
        activated_components (tuple, optional): _description_. Defaults to ("blocks.5.hook_mlp_out",).

    Returns:
        _type_: _description_
    """
    # 1. Deactivate context neuron, cache activations
    with model.hooks(fwd_hooks=context_ablation_hooks):
        _, ablated_cache = model.run_with_cache(prompt)

    # 2. Activate context neuron, ablate deactivated_components, cache activations    
    def deactivate_components_hook(value, hook: HookPoint):
        value = ablated_cache[hook.name]
        return value             
    deactivate_components_hooks = [(freeze_act_name, deactivate_components_hook) for freeze_act_name in deactivated_components]
    with model.hooks(fwd_hooks=deactivate_components_hooks+context_activation_hooks):
        _, context_and_activated_cache = model.run_with_cache(prompt)

    # 3. Deactivate context neuron, patch context neuron into activated_components, deactivate deactivated_components (doesn't matter when looking at MLP5)
    def activate_components_hook(value, hook: HookPoint):
        value = context_and_activated_cache[hook.name]
        return value         
    activate_components_hooks = [(freeze_act_name, activate_components_hook) for freeze_act_name in activated_components]
    with model.hooks(fwd_hooks=activate_components_hooks+context_ablation_hooks+deactivate_components_hooks):
        _, only_activated_cache = model.run_with_cache(prompt)
    return only_activated_cache

def clean_print_strings_as_html(strings: list[str], color_values: list[float], max_value: float=None, additional_measures: list[list[float]] | None = None, additional_measure_names: list[str] | None = None):
    """ Magic GPT function that prints a string as HTML and colors it according to a list of color values. Color values are normalized to the max value preserving the sign.
    """

    def normalize(values, max_value=None, min_value=None):
        if max_value is None:
            max_value = max(values)
        if min_value is None:
            min_value = min(values)
        min_value = abs(min_value)
        normalized = [value / max_value if value > 0 else value / min_value for value in values]
        return normalized
    
    html = "<div>"
    
    # Normalize color values
    normalized_values = normalize(color_values, max_value, max_value)

    for i in range(len(strings)):
        color = color_values[i]
        normalized_color = normalized_values[i]
        
        if color < 0:
            red = int(max(0, min(255, (1 + normalized_color) * 255)))
            green = int(max(0, min(255, (1 + normalized_color) * 255)))
            blue = 255
        else:
            red = 255
            green = int(max(0, min(255, (1 - normalized_color) * 255)))
            blue = int(max(0, min(255, (1 - normalized_color) * 255)))
        
        # Calculate luminance to determine if text should be black
        luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255
        
        # Determine text color based on background luminance
        text_color = "black" if luminance > 0.5 else "white"

        #visible_string = re.sub(r'\s+', '&nbsp;', strings[i])
        visible_string = re.sub(r'\s+', '_', strings[i])
        # visible_string = re.sub(r"[\n\t\s\r]*", "", visible_string)
        
        html += f'<span style="background-color: rgb({red}, {green}, {blue}); color: {text_color}; padding: 2px;" '
        html += f'title="Difference: {color_values[i]:.4f}' 
        if additional_measure_names is not None:
            for j in range(len(additional_measure_names)):
                html += f', {additional_measure_names[j]}: {additional_measures[j][i]:.4f}'
        html += f'">{visible_string}</span>'
    html += '</div>'

    # Print the HTML in Jupyter Notebook
    display(HTML(html))


def get_average_loss_plot_method(activate_context_fwd_hooks, deactivate_context_fwd_hooks, activated_component_name="MLP5",
                                deactivated_components = ("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out"),
                                activated_components = ("blocks.5.hook_mlp_out", ), plot=True, return_type="loss", answer_token=None):
    """Factory method that predefines the average_loss_plot variables which are constant within each notebook"""
    def average_loss_plot(prompts: list[str], model: HookedTransformer, token="", plot=plot):

        original_losses, ablated_losses, context_and_activated_losses, only_activated_losses = [], [], [], []
        names = ["Original", "Ablated", f"Context + {activated_component_name} active", f"{activated_component_name} active"]
        for prompt in prompts:
            pos = -1 if return_type == "loss" else -2
            original_loss, ablated_loss, context_and_activated_loss, only_activated_loss = \
                get_direct_effect(prompt, model, pos=pos, 
                                    context_ablation_hooks=deactivate_context_fwd_hooks, 
                                    context_activation_hooks=activate_context_fwd_hooks,
                                    deactivated_components=deactivated_components,
                                    activated_components=activated_components, return_type=return_type)
            if return_type=="logits":
                original_loss = original_loss[answer_token].item()
                ablated_loss = ablated_loss[answer_token].item()
                context_and_activated_loss = context_and_activated_loss[answer_token].item()
                only_activated_loss = only_activated_loss[answer_token].item()
            original_losses.append(original_loss)
            ablated_losses.append(ablated_loss)
            context_and_activated_losses.append(context_and_activated_loss)
            only_activated_losses.append(only_activated_loss)
        if plot:
            plot_barplot([original_losses, ablated_losses, context_and_activated_losses, only_activated_losses], names, ylabel="Loss", title=f"Average loss '{token}'")
        else:
            return original_losses, ablated_losses, context_and_activated_losses, only_activated_losses
    return average_loss_plot

def get_common_tokens(data, model, ignore_tokens, k=100) -> Tensor:
    # Get top common german tokens excluding punctuation
    token_counts = torch.zeros(model.cfg.d_vocab).cuda()
    for example in tqdm(data):
        tokens = model.to_tokens(example)
        for token in tokens[0]:
            token_counts[token.item()] += 1

    punctuation = ["\n", ".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\", "\"", "'"]
    leading_space_punctuation = [" " + char for char in punctuation]
    punctuation_tokens = model.to_tokens(punctuation + leading_space_punctuation + [' –', " ", '  ', "<|endoftext|>"])[:, 1].flatten()
    token_counts[punctuation_tokens] = 0
    token_counts[ignore_tokens] = 0

    top_counts, top_tokens = torch.topk(token_counts, k)
    return top_tokens

def get_random_selection(tensor, n):
        indices = torch.randint(0, len(tensor), (n,))
        return tensor[indices]

def generate_random_prompts(end_string, model, random_tokens, n=50, length=12):
    # Generate a batch of random prompts ending with a specific ngram
    end_tokens = model.to_tokens(end_string).flatten()[1:]
    prompts = []
    for i in range(n):
        prompt = get_random_selection(random_tokens, n=length).cuda()
        prompt = torch.cat([prompt, end_tokens])
        prompts.append(prompt)
    prompts = torch.stack(prompts)
    return prompts

def replace_column(prompts: Int[Tensor, "n_prompts n_tokens"], token_index: int, replacement_tokens: Int[Tensor, "n_tokens"]):
    # Replaces a specific token position in a batch of prompts with random common German tokens
    new_prompts = prompts.clone()
    random_tokens = get_random_selection(replacement_tokens, n=prompts.shape[0]).cuda()
    new_prompts[:, token_index] = random_tokens
    return new_prompts 

def get_context_ablation_hooks(layer_to_ablate: int, neurons_to_ablate: list[int], model: HookedTransformer):
    german_data = load_json_data("data/german_europarl.json")[:200]
    english_data = load_json_data("data/english_europarl.json")[:200]

    english_activations = {}
    german_activations = {}
    
    english_activations[layer_to_ablate] = get_mlp_activations(english_data, layer_to_ablate, model, mean=False)
    german_activations[layer_to_ablate] = get_mlp_activations(german_data, layer_to_ablate, model, mean=False)

    MEAN_ACTIVATION_ACTIVE = german_activations[layer_to_ablate][:, neurons_to_ablate].mean()
    MEAN_ACTIVATION_INACTIVE = english_activations[layer_to_ablate][:, neurons_to_ablate].mean()

    def deactivate_neurons_hook(value, hook):
        value[:, :, neurons_to_ablate] = MEAN_ACTIVATION_INACTIVE
        return value
    deactivate_neurons_fwd_hooks=[(f'blocks.{layer_to_ablate}.mlp.hook_post', deactivate_neurons_hook)]

    def activate_neurons_hook(value, hook):
        value[:, :, neurons_to_ablate] = MEAN_ACTIVATION_ACTIVE
        return value
    activate_neurons_fwd_hooks=[(f'blocks.{layer_to_ablate}.mlp.hook_post', activate_neurons_hook)]

    return activate_neurons_fwd_hooks, deactivate_neurons_fwd_hooks

def get_ablate_attention_hook(layer, head, mean_activation, pos: int | None=-2):
    def ablate_attention_head(value, hook):
        if pos is None:
            value[:, :, head, :] = mean_activation[layer, head, :]
        else:
            value[:, pos, head, :] = mean_activation[layer, head, :]
        return value

    return (f'blocks.{layer}.attn.hook_z', ablate_attention_head)

def get_trigram_prompts(prompts, first_replacement_tokens, second_replacement_tokens):
    # Create prev-random prompts
    test_prompts_prev = replace_column(prompts, -2, second_replacement_tokens)
    # Create random-random prompts matching random tokens from previous prompts
    random_prompts_prev = replace_column(test_prompts_prev, -3, first_replacement_tokens)

    # Create random-current prompts
    test_prompts_curr = replace_column(prompts, -3, first_replacement_tokens)
    # Create random-random prompts matching random tokens from previous prompts
    random_prompts_curr = replace_column(test_prompts_curr, -2, second_replacement_tokens)
    return prompts, test_prompts_prev, random_prompts_prev, test_prompts_curr, random_prompts_curr

def get_residual_trigram_directions(prompt_tuple, model, layer, ):

    _, test_prompts_prev, random_prompts_prev, test_prompts_curr, random_prompts_curr = prompt_tuple

    _, test_cache_prev = model.run_with_cache(test_prompts_prev)
    res_cache_prev = test_cache_prev[f'blocks.{layer}.hook_resid_post'][:, -2]
    _, random_cache_prev = model.run_with_cache(random_prompts_prev)
    res_cache_random_prev = random_cache_prev[f'blocks.{layer}.hook_resid_post'][:, -2]

    _, test_cache_curr = model.run_with_cache(test_prompts_curr)
    res_cache_curr = test_cache_curr[f'blocks.{layer}.hook_resid_post'][:, -2]
    _, random_cache_curr = model.run_with_cache(random_prompts_curr)
    res_cache_random_curr = random_cache_curr[f'blocks.{layer}.hook_resid_post'][:, -2]

    prev_token_direction = (res_cache_prev - res_cache_random_prev).mean(0)
    curr_token_direction = (res_cache_curr - res_cache_random_curr).mean(0)

    return prev_token_direction, curr_token_direction

def get_trigram_neuron_activations(prompt_tuple, model, deactivate_neurons_fwd_hooks, layer=5, mlp_hook="hook_pre"):

    cache_name = F"blocks.{layer}.mlp.{mlp_hook}"
    neurons = torch.LongTensor([i for i in range(2048)])
    neuron_list = neurons.tolist()

    data = {
        "Neuron": [], 
        "PrevTokenPresent": [], 
        "CurrTokenPresent": [],
        "ContextPresent": [],
        "Activation": [],
        }

    def add_to_data(data: dict[str, list], cache_result: Tensor, neurons, prev_token: bool, curr_token: bool, context_neuron: bool):
        data["Neuron"].extend(neuron_list)
        data["PrevTokenPresent"].extend([prev_token] * len(neurons))
        data["CurrTokenPresent"].extend([curr_token] * len(neurons))
        data["ContextPresent"].extend([context_neuron] * len(neurons))
        data["Activation"].extend(cache_result[neurons].tolist())
        return data

    def get_mean_activations(prompts):
        with model.hooks([(cache_name, save_activation)]):
            model.run_with_hooks(prompts)
            act_original = model.hook_dict[cache_name].ctx['activation']
            act_original = act_original[:, -2].mean(0)

        with model.hooks(fwd_hooks=deactivate_neurons_fwd_hooks+[(cache_name, save_activation)]):
            model.run_with_hooks(prompts)
            act_ablated = model.hook_dict[cache_name].ctx['activation']
            act_ablated = act_ablated[:, -2].mean(0)
        return act_original, act_ablated

    full_prompts, test_prompts_prev, random_prompts_prev, test_prompts_curr, random_prompts_curr = prompt_tuple

    # Full prompt
    act_original, act_ablated = get_mean_activations(full_prompts)
    data = add_to_data(data, act_original, neurons, prev_token=True, curr_token=True, context_neuron=True)
    data = add_to_data(data, act_ablated, neurons, prev_token=True, curr_token=True, context_neuron=False)

    # 1st token present prompts
    act_original, act_ablated = get_mean_activations(test_prompts_prev)
    data = add_to_data(data, act_original, neurons, prev_token=True, curr_token=False, context_neuron=True)
    data = add_to_data(data, act_ablated, neurons, prev_token=True, curr_token=False, context_neuron=False)

    # 2nd token present prompts
    act_original, act_ablated = get_mean_activations(test_prompts_curr)
    data = add_to_data(data, act_original, neurons, prev_token=False, curr_token=True, context_neuron=True)
    data = add_to_data(data, act_ablated, neurons, prev_token=False, curr_token=True, context_neuron=False)

    # No token present 
    act_original_1, act_ablated_1 = get_mean_activations(random_prompts_prev)
    act_original_2, act_ablated_2 = get_mean_activations(random_prompts_curr)
    act_original = (act_original_1 + act_original_2) / 2
    act_ablated = (act_ablated_1 + act_ablated_2) / 2
    data = add_to_data(data, act_original, neurons, prev_token=False, curr_token=False, context_neuron=True)
    data = add_to_data(data, act_ablated, neurons, prev_token=False, curr_token=False, context_neuron=False)

    df = pd.DataFrame(data)

    # Pivot the dataframe
    df['Prev/Curr/Context'] = df.apply(lambda row: ("Y" if row['PrevTokenPresent'] else "N")+("Y"if row['CurrTokenPresent'] else "N")+("Y" if row['ContextPresent'] else "N"), axis=1)
    pivot_df = df.pivot(index='Neuron', columns='Prev/Curr/Context', values='Activation')
    return pivot_df

def union_where(tensors: list[Float[Tensor, "n"]], cutoff: float, greater: bool = True) -> Int[Tensor, "m"]:
    tensor_length = tensors[0].shape[0]
    union = np.array(range(tensor_length))
    for tensor in tensors:
        if greater:
            indices = torch.where(tensor > cutoff)[0]
        else:
            indices = torch.where(tensor < cutoff)[0]
        union = np.intersect1d(union, indices.cpu().numpy())
    return torch.LongTensor(union).cuda()


def get_boosted_tokens(prompts, model, ablation_hooks: list, all_ignore: Float[Tensor, "n"], logit_pos=-2, threshold=-7, deboost=False, log=True):
    original_logprobs = model(prompts, return_type="logits", loss_per_token=True).log_softmax(-1)[:, logit_pos]
    with model.hooks(fwd_hooks=ablation_hooks):
        deactivated_logprobs = model(prompts, return_type="logits", loss_per_token=True).log_softmax(-1)[:, logit_pos]
    
    if deboost:
        logprob_diffs = (deactivated_logprobs - original_logprobs).mean(0)
    else: 
        logprob_diffs = (original_logprobs - deactivated_logprobs).mean(0)

    logprob_diffs[original_logprobs.mean(0) < threshold] = 0
    logprob_diffs[all_ignore] = 0

    top_diffs, top_tokens = torch.topk(logprob_diffs, model.cfg.d_vocab)
    num_meaningful_diffs = (top_diffs > 0).sum().item()

    boost_amounts = top_diffs[:num_meaningful_diffs]
    boosted_tokens = model.to_str_tokens(top_tokens[:num_meaningful_diffs])
    if log:
        boost_str = "Boosted" if not deboost else "Deboosted"
        sign = "+" if not deboost else "-"
        token_str = [f"{token} ({sign}{boost:.2f})" for token, boost in zip(boosted_tokens, boost_amounts)]
        print(f"{boost_str} tokens: " + ", ".join(token_str))
    else:
        return top_diffs[:num_meaningful_diffs], top_tokens[:num_meaningful_diffs]

def create_ablation_prompts(prompts, ablation_mode, common_tokens):
    # Simplifies column replacement by applying it to different ablation modes
    if ablation_mode[0] == "N":
        prompts = replace_column(prompts, -3, common_tokens)
    if ablation_mode[1] == "N":
        prompts = replace_column(prompts, -2, common_tokens)
    return prompts

def compute_mlp_loss(prompts, model, df, neurons, ablate_mode="NNN", layer=5, compute_original_loss=False):

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

def get_top_k_neurons(df, condition, sortby, k=10, ascending=False):
    tmp_df = df[condition].copy()
    tmp_df = tmp_df.sort_values(by=sortby, ascending=ascending)
    return tmp_df.index[:k]

def split_tokens_with_space(tokens: Tensor, model: HookedTransformer) -> tuple[Tensor, Tensor]:
    new_word_tokens = []
    continuation_tokens = []
    for token in tokens:
        str_token = model.to_single_str_token(token.item())
        if str_token.startswith(" "):
            new_word_tokens.append(token)
        else:
            continuation_tokens.append(token)
    new_word_tokens = torch.stack(new_word_tokens)
    continuation_tokens = torch.stack(continuation_tokens)
    return new_word_tokens, continuation_tokens

def activation_data_frame(ngram: str, prompts: Tensor, model: HookedTransformer, common_tokens: Tensor, deactivate_neurons_fwd_hooks: list[tuple[str, Callable]], layer=5, mlp_hook="hook_pre") -> pd.DataFrame:
    new_word_tokens, continuation_tokens = split_tokens_with_space(common_tokens, model)
    if ngram.startswith(" "):
        prompt_tuple = get_trigram_prompts(prompts, new_word_tokens, continuation_tokens)
    else:
        prompt_tuple = get_trigram_prompts(prompts, continuation_tokens, continuation_tokens)

    df = get_trigram_neuron_activations(prompt_tuple, model, deactivate_neurons_fwd_hooks ,layer, mlp_hook=mlp_hook)
    return df

def get_mean_neuron_activations(neurons: list[(int, int)], data: list[str], model: HookedTransformer):
    # Neuron is a tuple of (layer, neuron)
    layers = list(set([x[0] for x in neurons]))
    activations = {}
    for layer in layers:
        activations[layer] = get_mlp_activations(data, layer, model, mean=True)
    mean_activations = []
    for layer, neuron in neurons:
        mean_activations.append(activations[layer][neuron])
    return mean_activations

    