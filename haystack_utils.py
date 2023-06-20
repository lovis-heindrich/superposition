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

def get_caches_single_prompt(prompt: str, model: HookedTransformer, mean_neuron_activations: Float[Tensor, "d_mlp"], neurons =(609), layer_to_ablate=3) -> tuple[float, float, ActivationCache, ActivationCache]:
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
    neurons = torch.LongTensor(neurons)
    def ablate_neuron_hook(value, hook):
        value[:, :, neurons] = mean_neuron_activations[neurons]
        return value
    
    tokens = model.to_tokens(prompt)
    original_loss, original_cache = model.run_with_cache(tokens, return_type="loss")

    with model.hooks(fwd_hooks=[(f'blocks.{layer_to_ablate}.mlp.hook_post', ablate_neuron_hook)]):
        ablated_loss, ablated_cache = model.run_with_cache(tokens, return_type="loss")

    return original_loss.item(), ablated_loss.item(), original_cache, ablated_cache