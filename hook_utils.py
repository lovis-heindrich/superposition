from typing import Literal
import torch
from jaxtyping import Int, Float
from torch import Tensor
from collections import defaultdict


def save_activation(value, hook):
    """
    Add a hook using this method and an activation label then use the label to access the activation:
    act = model.hook_dict[act_label].ctx['activation'][:, :, 669]
    """
    hook.ctx['activation'] = value #.detach().cpu().to(torch.float16)
    return value


def get_ablate_neuron_hook(layer: int, neuron: int | Int[Tensor, "n"], act_value: Float[Tensor, "n"] | float, hook_point: Literal["post", "pre"]="post"):
    def neuron_hook(value, hook):
        value[:, :, neuron] = act_value
        return value
    return (f'blocks.{layer}.mlp.hook_{hook_point}', neuron_hook)


def get_ablate_context_neurons_hooks(context_neurons: list[tuple[int, int]], activations: list[float]):
    hooks = []
    # Group neurons by layer
    per_layer_neurons = defaultdict(list)
    for i, context_neuron in enumerate(context_neurons):
        layer, neuron = context_neuron
        activation = activations[i]
        per_layer_neurons[layer].append((neuron, activation))

    # Create hooks per layer
    for layer, layer_data in per_layer_neurons.items():
        neurons, activations = zip(*layer_data)
        neurons = torch.LongTensor(neurons).cuda()
        activations = torch.Tensor(activations).cuda()
        hooks.append(get_ablate_neuron_hook(layer, neurons, activations))
    return hooks