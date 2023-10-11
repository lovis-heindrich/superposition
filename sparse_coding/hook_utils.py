from typing import Literal
import torch
from jaxtyping import Int, Float
from torch import Tensor
from collections import defaultdict
import haystack_utils

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


def get_ablate_neurons_hook(neuron: int | list[int], ablated_cache, layer=5, hook_point="hook_post"):
    def ablate_neurons_hook(value, hook):
        value[:, :, neuron] = ablated_cache[f'blocks.{layer}.mlp.{hook_point}'][:, :, neuron]
        return value
    return [(f'blocks.{layer}.mlp.{hook_point}', ablate_neurons_hook)]

import numpy as np
def get_resample_neurons_hooks(neurons: list[tuple[int, int]], resampled_cache, hook_point="hook_post"):
    layer_neurons = haystack_utils.get_neurons_by_layer(neurons)
    hooks = []
    for layer, neurons in layer_neurons.items():
        hook_name = f'blocks.{layer}.mlp.{hook_point}'
        def resample_neurons_hook(value, hook):
            resample_cache_slice = np.s_[:value.shape[0], :value.shape[1], neurons]
            value[:, :, neurons] = resampled_cache[hook_name][resample_cache_slice]
            return value
        hooks.append((hook_name, resample_neurons_hook))
    return hooks


def get_snap_to_peak_1_hook():
    def snap_to_peak_1(value, hook):
        '''Doesn't snap disabled and ambiguous activations'''
        neuron_act = value[:, :, 2994]
        value[:, :, 2994][(neuron_act > 0.8) & (neuron_act < 4.1)] = 3.5
        value[:, :, 2994][(neuron_act > 4.8)] = 3.5
        return value
    return ('blocks.8.mlp.hook_post', snap_to_peak_1)

def get_snap_to_peak_2_hook():
    def snap_to_peak_2(value, hook):
        '''Doesn't snap disabled and ambiguous activations'''
        neuron_act = value[:, :, 2994]
        value[:, :, 2994][(neuron_act > 0.8) & (neuron_act < 4.1)] = 5.5
        value[:, :, 2994][(neuron_act > 4.8)] = 5.5
        return value
    return ('blocks.8.mlp.hook_post', snap_to_peak_2)