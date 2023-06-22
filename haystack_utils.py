from transformer_lens import HookedTransformer, ActivationCache
from jaxtyping import Float
from torch import Tensor
import einops
from tqdm.auto import tqdm
import torch
from typing import List
import plotly.express as px
import gc

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


def get_head_activations(
    prompts: List[str],
    layer: int,
    model: HookedTransformer,
    num_prompts: int = -1,
    context_crop_start=10,
    context_crop_end=400,
    mean=True
):
    """
    Runs the model through a list of prompts and stores the head patterns for a given layer. 
    The mean calculation is somewhat complex because each head has a different attention matrix size.
    We pad the patterns with zeros to the maximum context length so they can be stacked. 
    We saves a running count of attention head size to allow for mean pooling. To use the counts,
    we convert it to a matrix where [0, 0] is the first position count, [0, 1], [1, 0], and [1, 1] all 
    have the second position count, and so on. Then we can do elementwise division over the summed
    attention pattern to get the mean.

    A different implementation is done for MLPs in Haystack_cleaned using a mask.
    """
    max_ctx = context_crop_end - context_crop_start
    position_counts = torch.zeros(max_ctx).cuda()
    patterns = []
    pattern_label = f'blocks.{layer}.attn.hook_pattern'
    if num_prompts == -1:
        num_prompts = len(prompts)
    for i in tqdm(range(num_prompts)):
        tokens = model.to_tokens(prompts[i])
        _, cache = model.run_with_cache(tokens)
        # cache[pattern_label].shape == [batch head query_pos key_pos]
        pattern = cache[pattern_label][:, :, context_crop_start:context_crop_end, context_crop_start:context_crop_end]
        patterns.append(pattern)
        position_counts[:pattern.shape[2]] += 1
    
    padded_patterns = []
    for pattern in patterns:
        pad = (0, max_ctx - pattern.shape[-1], 0, max_ctx - pattern.shape[-1])
        padded_patterns.append(torch.nn.functional.pad(pattern, pad))
    
    patterns = torch.concat(padded_patterns, dim=0)
    if mean:
        scaling_matrix = torch.zeros(max_ctx, max_ctx).cuda()
        for row in range(max_ctx):
            for col in range(max_ctx):
                scaling_matrix[row, col] = position_counts[max(row, col)]

        print(scaling_matrix.shape)
        print(patterns.shape)
        return patterns.sum(dim=0) / scaling_matrix # [num_prompts head max_pos_len max_pos_len] 
    return patterns


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

def get_caches_single_prompt(prompt: str, model: HookedTransformer, mean_neuron_activations: Float[Tensor, "d_mlp"], neurons =(609), layer_to_ablate=3, crop_context: None | tuple[int, int]=None, return_type: str = "loss") -> tuple[float, float, ActivationCache, ActivationCache]:
    """ Runs the model with and without ablation on a single prompt and returns the caches.

    Args:
        prompt (str): Prompt to run.
        model (HookedTransformer): Model to run.
        mean_neuron_activations (Float[Tensor, "d_mlp"]): Mean neuron activations for the layer to ablate.
        neurons (tuple, optional): Neurons to ablate. Defaults to [609].
        layer_to_ablate (int, optional): Layer to ablate. Defaults to 3.

    Returns:
        tuple[float, float, ActivationCache, ActivationCache]: Original loss, ablated loss, original cache, ablated cache.
    """
    assert return_type in ["loss", "logits"]
    neurons = torch.LongTensor(neurons)
    def ablate_neuron_hook(value, hook):
        value[:, :, neurons] = mean_neuron_activations[neurons]
        return value
    
    if crop_context is not None:
        tokens = model.to_tokens(prompt)[:, crop_context[0]:crop_context[1]]
    else:
        tokens = model.to_tokens(prompt)
  
    original_return_value, original_cache = model.run_with_cache(tokens, return_type=return_type)

    with model.hooks(fwd_hooks=[(f'blocks.{layer_to_ablate}.mlp.hook_post', ablate_neuron_hook)]):
        ablated_return_value, ablated_cache = model.run_with_cache(tokens, return_type=return_type)
    
    if return_type == "loss":
        return original_return_value.item(), ablated_return_value.item(), original_cache, ablated_cache
    else:
        return original_return_value, ablated_return_value, original_cache, ablated_cache


def get_average_loss(data: list[str], model:HookedTransformer, batch_size=1, crop_context=-1, fwd_hooks=[], positionwise=False):
    if crop_context == -1:
        crop_context = model.cfg.n_ctx

    position_counts = torch.zeros(model.cfg.n_ctx).cuda()
    position_loss = torch.zeros(model.cfg.n_ctx).cuda()

    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    for batch in dataloader:
        tokens = model.to_tokens(batch)[:, :crop_context].cuda()
        loss = model.run_with_hooks(tokens, return_type="loss", loss_per_token=True, fwd_hooks=fwd_hooks) # includes BOS, excludes final token of longest prompt in batch
    
        # Produce a mask of every token which we expect to produce a valid loss value. Excludes padding tokens and the final token of each row.
        active_tokens = tokens[:, :] != model.tokenizer.pad_token_id
        active_tokens[:, 0] = True  # BOS token ID is sometimes equivalent to pad token ID so we manually set it to true
        # Set each final token prediction to False
        last_true_token_mask = (active_tokens.flip(dims=[1]).cumsum(dim=1) == 1).flip(dims=[1])
        last_true_indices = last_true_token_mask.nonzero() # Tensor of tuples of each final True index in the batch
        active_tokens[last_true_indices[:, 0], last_true_indices[:, 1]] = False
        # Remove the final column which will never have an associated loss. The mask size is now equal to the loss tensor size.
        print(active_tokens.shape)
        active_tokens = active_tokens[:, :-1] 
        print(active_tokens.shape)


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

    print(position_loss[:10])
    print(position_counts[:10])
    return position_loss.sum() / position_counts.sum()


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
