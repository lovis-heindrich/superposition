from collections import defaultdict

import torch

from haystack_utils import load_txt_data, get_mlp_activations

class GermanHook():
    # english_neurons = [(5, 395), (5, 166), (5, 908), (5, 285), (3, 862), (5, 73), (4, 896), (5, 348), (5, 297), (3, 1204)]
    german_neurons = [(4, 482), (5, 1039), (5, 407), (5, 1516), (5, 1336), (4, 326), (5, 250), (3, 669)]
    # french_neurons = [(5, 112), (4, 1080), (5, 1293), (5, 455), (5, 5), (5, 1901), (5, 486), (4, 975)]
    # German neurons that still score highly on our dataset
    our_german_neurons = [(3, 669), (5, 1336), (4, 482), (5, 1039), (4, 326)]
    
    def __init__(self, model):
        """Language neuron hooks for Pythia 70m V1 """
        
        # Load data
        english_data = load_txt_data("data/kde4_english.txt")[:500]
        german_data = load_txt_data("data/wmt_german_large.txt")[:500]
        # french_data = load_txt_data("data/kde4_french.txt")[:500]

        # Define hook patterns
        self.mlp_pattern = lambda name: name.endswith("mlp.hook_post")

        # English neurons
        self.english_mean_high_activations = defaultdict(torch.Tensor, {
            3: get_mlp_activations(english_data, 3, model, mean=True),  # [2048]
            4: get_mlp_activations(english_data, 4, model, mean=True),
            5: get_mlp_activations(english_data, 5, model, mean=True)
        })
        self.english_mean_low_activations = defaultdict(torch.Tensor, {
            3: get_mlp_activations(german_data, 3, model, mean=True),  # [2048]
            4: get_mlp_activations(german_data, 4, model, mean=True),
            5: get_mlp_activations(german_data, 5, model, mean=True)
        })
        
        # French neurons
        # self.french_mean_high_activations = defaultdict(torch.Tensor, {
        #     4: get_mlp_activations(french_data, 4, model, mean=True),  # [2048]
        #     5: get_mlp_activations(french_data, 5, model, mean=True)
        # })
        # self.french_mean_low_activations = defaultdict(torch.Tensor, {
        #     4: get_mlp_activations(english_data, 4, model, mean=True),
        #     5: get_mlp_activations(english_data, 5, model, mean=True)
        # })

        # German neurons
        self.german_mean_high_activations = defaultdict(torch.Tensor, {
            3: get_mlp_activations(german_data, 3, model, mean=True),  # [2048]
            4: get_mlp_activations(german_data, 4, model, mean=True),
            5: get_mlp_activations(german_data, 5, model, mean=True)
        })
        self.german_mean_low_activations = defaultdict(torch.Tensor, {
            3: get_mlp_activations(english_data, 3, model, mean=True),  # [2048]
            4: get_mlp_activations(english_data, 4, model, mean=True),
            5: get_mlp_activations(english_data, 5, model, mean=True)
        })
        self.our_german_neurons_by_layer = defaultdict(list)
        for item in GermanHook.our_german_neurons:
            self.our_german_neurons_by_layer[item[0]].append(item[1])
        self.german_neurons_by_layer = defaultdict(list)
        for item in GermanHook.german_neurons:
            self.german_neurons_by_layer[item[0]].append(item[1])


    def get_disable_german_subset_hook(self):
        def hook(value, hook):
            layer = hook.layer()
            german_neurons_for_layer = self.our_german_neurons_by_layer[layer]
            value[:, :, german_neurons_for_layer] = self.german_mean_low_activations[layer][german_neurons_for_layer].cuda() * 2.2
            return value
        return hook

    def get_disable_german_hook(self):
        def hook(value, hook):
            layer = hook.layer()
            german_neurons_for_layer = self.german_neurons_by_layer[layer]
            value[:, :, german_neurons_for_layer] = self.german_mean_low_activations[layer][german_neurons_for_layer].cuda() * 2.2
            return value
        return hook

    def get_enable_german_hook(self):
        def hook(value, hook):
            layer = hook.layer()
            german_neurons_for_layer = self.german_neurons_by_layer[layer]
            value[:, :, german_neurons_for_layer] = self.german_mean_high_activations[layer][german_neurons_for_layer].cuda() * 2.2
            return value
        return hook

    def get_disable_german_l3_hook(self):
        def hook(value, hook):
            if hook.layer() == 3:
                value[:, :, 619] = self.german_mean_low_activations[3][619].cuda() * 2.2
            return value
        return hook

    def get_enable_german_l3_hook(self):
        def hook(value, hook):
            if hook.layer() == 3:
                value[:, :, 619] = self.german_mean_high_activations[3][619].cuda() * 2.2
            return value
        return hook

