from transformer_lens import HookedTransformer, ActivationCache
from jaxtyping import Float
from torch import Tensor
import einops
from tqdm.auto import tqdm
import torch
from typing import List, Tuple
import plotly.express as px
import plotly.graph_objects as go
import gc
import numpy as np


def get_mlp_activations(
    prompts: List[str],
    layer: int,
    model: HookedTransformer,
    num_prompts: int = -1,
    context_crop_start=10,
    context_crop_end=400,
    mean=True
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



def get_average_loss(data: list[str], model: HookedTransformer, crop_context=-1, fwd_hooks=[], positionwise=False):
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


def get_loss_increase_for_component(prompts: list[str], model: HookedTransformer, fwd_hooks=[], patched_component=8, crop_context_end: None | int=None, disable_progress_bar=False):
    original_losses = []
    patched_losses = []
    for prompt in tqdm(prompts, disable=disable_progress_bar):
        if crop_context_end is not None:
            tokens = model.to_tokens(prompt)[:, :crop_context_end]
        else:
            tokens = model.to_tokens(prompt)

        original_loss, original_cache = model.run_with_cache(tokens, return_type="loss")
        with model.hooks(fwd_hooks=fwd_hooks):
            ablated_loss, ablated_cache = model.run_with_cache(tokens, return_type="loss")

        # component, batch, pos, residual
        # TODO figure out if we need layer norm here
        original_per_layer_residual, original_labels = original_cache.decompose_resid(layer=-1, return_labels=True, apply_ln=False)
        ablated_per_layer_residual, ablated_labels = ablated_cache.decompose_resid(layer=-1, return_labels=True, apply_ln=False)

        # ['embed', '0_attn_out', '0_mlp_out', '1_attn_out', '1_mlp_out', '2_attn_out', '2_mlp_out', '3_attn_out', '3_mlp_out', '4_attn_out', '4_mlp_out', '5_attn_out', '5_mlp_out']
        def swap_cache_hook(value, hook):
            # Batch, pos, residual
            value -= original_per_layer_residual[patched_component]
            value += ablated_per_layer_residual[patched_component]
        
        with model.hooks(fwd_hooks=[(f'blocks.5.hook_resid_post', swap_cache_hook)]):
            patched_loss = model(tokens, return_type="loss")

        original_losses.append(original_loss.item())
        patched_losses.append(patched_loss.item())


    print(f"Original loss: {np.mean(original_losses):.2f}, patched loss: {np.mean(patched_losses):.2f} (+{((np.mean(patched_losses) - np.mean(original_losses)) / np.mean(original_losses))*100:.2f}%)")
    return np.mean(original_losses), np.mean(patched_losses)


def get_ablated_performance(data: list[str], model: HookedTransformer, fwd_hooks: List[Tuple]=[], batch_size=1, display_tqdm=True):
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
) -> tuple[float, float, ActivationCache, ActivationCache]:
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


def two_histogram(data_1: Float[Tensor, "n"], data_2: Float[Tensor, "n"], data_1_name="", data_2_name="", title: str = "", x_label: str= "", y_label: str=""):
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


def line(x, xlabel="", ylabel="", title="", xticks=None, width=800):
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
