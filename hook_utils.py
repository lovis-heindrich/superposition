def get_ablate_mlp_neurons_hook(neurons, layer, cache):
    def ablate_neurons_hook(value, hook):
        value[:, :, neurons] = cache[f'blocks.{layer}.mlp.hook_post'][:, :, neurons].mean((0, 1))
        return value
    return [(f'blocks.{layer}.mlp.hook_post', ablate_neurons_hook)]