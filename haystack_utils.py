import warnings
from transformer_lens import HookedTransformer, ActivationCache, utils
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float, Int
from torch import Tensor
import einops
from tqdm.auto import tqdm
import torch
from typing import List, Tuple
import plotly.express as px
import plotly.graph_objects as go
import gc
import numpy as np
from einops import einsum
from IPython.display import display, HTML
import re



def DLA(prompts: List[str], model: HookedTransformer):
    logit_attributions = []
    for prompt in tqdm(prompts):
        tokens = model.to_tokens(prompt)
        answers = tokens[:, 1:]
        tokens = tokens[:, :-1]
        answer_residual_directions = model.tokens_to_residual_directions(answers)  # [batch pos d_model]
        _, cache = model.run_with_cache(tokens)
        accumulated_residual, labels = cache.accumulated_resid(layer=-1, incl_mid=False, pos_slice=None, return_labels=True)
        # Component batch pos d_model
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
    hook_pre = False
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
    if hook_pre:
        act_label = f"blocks.{layer}.mlp.hook_pre"
    else:
        act_label = f"blocks.{layer}.mlp.hook_post"
    if num_prompts == -1:
        num_prompts = len(prompts)
    for i in tqdm(range(num_prompts)):
        tokens = model.to_tokens(prompts[i])
        _, cache = model.run_with_cache(tokens)
        act = cache[act_label][:, context_crop_start:context_crop_end, :]
        act = einops.rearrange(act, "batch pos d_mlp -> (batch pos) d_mlp")
        acts.append(act)
    acts = torch.concat(acts, dim=0)
    if mean:
        return torch.mean(acts, dim=0)
    return acts


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


def line(x, xlabel="", ylabel="", title="", xticks=None, width=800, hover_data=None):
    fig = px.line(x, title=title)
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, width=width)
    if xticks != None:
        fig.update_layout(
            xaxis = dict(
            tickmode = 'array',
            tickvals = [i for i in range(len(xticks))],
            ticktext = xticks
            )
        )
    if hover_data != None:
        fig.update(data=[{'customdata': hover_data, 'hovertemplate': "Loss: %{y:.4f} (+%{customdata:.2f}%)"}])
    fig.show()


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
    with open(path, "r") as f:
        data = f.read().split("\n")
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
        #print(f"Original loss: {np.max(original_loss):.2f}, frozen loss: {np.max(frozen_loss):.2f} (+{((np.mean(frozen_loss) - np.mean(original_loss)) / np.mean(original_losses))*100:.2f}%), ablated loss: {np.mean(ablated_losses):.2f} (+{((np.mean(ablated_losses) - np.mean(original_losses)) / np.mean(original_losses))*100:.2f}%)")
        return original_loss, total_effect_loss_change, direct_effect_loss_change, indirect_effect_loss_change
    

def plot_barplot(data: list[list[float]], names: list[str], xlabel="", ylabel="", title="", width=1000):
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)
    
    fig = go.Figure()
    
    for i in range(len(names)):
        fig.add_trace(go.Bar(
            x=[names[i]],
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
    )
    
    fig.show()


# New

def get_mlp5_attribution_without_mlp4(prompt: str, model: HookedTransformer, ablation_hooks: list, pos: int | None = -1):
    # Freeze everything except for MLP5 to see if MLP5 depends on MLP4
    freeze_act_names=("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out")
    original_loss, total_effect_loss, direct_mlp3_mlp5_loss, _= split_effects(prompt, model, ablation_hooks=ablation_hooks, freeze_act_names=freeze_act_names, debug_log=False, return_absolute=True)
    freeze_act_names=("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out", "blocks.5.hook_mlp_out")
    _, _, direct_mlp3_loss, _ = split_effects(prompt, model, ablation_hooks=ablation_hooks, freeze_act_names=freeze_act_names, debug_log=False, return_absolute=True)

    if pos is not None:
        return original_loss[0, pos].item(), total_effect_loss[0, pos].item(), direct_mlp3_mlp5_loss[0, pos].item(), direct_mlp3_loss[0, pos].item()
    else:
        return original_loss[0, :], total_effect_loss[0, :], direct_mlp3_mlp5_loss[0, :], direct_mlp3_loss[0, :]


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
        scaled_neuron_directions = cache.apply_ln_to_stack(neuron_directions)[:, 0, pos-1, :] # [neuron embed]
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
        answer_tokens = tokens[:, pos] #TODO check
    # Get difference between ablated and unablated neurons' contribution to answer logit
    _, _, original_cache, ablated_cache = get_caches_single_prompt(
        prompt, model, fwd_hooks)
    original_unembedded = get_neuron_logit_contribution(original_cache, model, answer_tokens, layer=layer_to_compare, pos=pos) # [neuron]
    ablated_unembedded = get_neuron_logit_contribution(ablated_cache, model, answer_tokens, layer=layer_to_compare, pos=pos)
    differences = (original_unembedded - ablated_unembedded).detach().cpu() # [neuron]
    return differences

def get_neuron_loss_attribution(prompt, model, neurons, ablation_hooks: list, pos: int | None=-1):
    original_loss, original_cache = model.run_with_cache(prompt, return_type="loss", loss_per_token=True)
    with model.hooks(fwd_hooks=ablation_hooks):
        ablated_loss, ablated_cache = model.run_with_cache(prompt, return_type="loss", loss_per_token=True)

    # Remove the effects of ablating at MLP3 from the components after MLP3
    def freeze_neurons_hook(value, hook: HookPoint):
        if pos is not None:
            value[:, :, neurons] = original_cache[hook.name][:, :, neurons] # [batch pos neuron]
        else:
            value_tmp = value.clone()[0, :-1, :]
            src_tmp = original_cache[hook.name][0, :-1, :].gather(dim=-1, index=neurons.cuda())
            tmp = value_tmp.scatter_(dim=-1, index=neurons.cuda(), src=src_tmp)
            value[0, :-1, :] = tmp
        return value      

    freeze_original_hooks = [("blocks.5.mlp.hook_post", freeze_neurons_hook)]
    with model.hooks(fwd_hooks=ablation_hooks+freeze_original_hooks):
        ablated_with_original_frozen_loss = model(prompt, return_type="loss", loss_per_token=True)
    #print(ablated_loss[0, :], ablated_with_original_frozen_loss[0, :])
    if pos is not None:
        return original_loss[0, pos].item(), ablated_loss[0, pos].item(), ablated_with_original_frozen_loss[0, pos].item()
    else:
        return original_loss[0, :], ablated_loss[0, :], ablated_with_original_frozen_loss[0, :]