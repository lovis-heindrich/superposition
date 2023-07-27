import torch


def get_ablate_mlp_neurons_hook(neurons, layer, cache):
    def ablate_neurons_hook(value, hook):
        value[:, :, neurons] = cache[f'blocks.{layer}.mlp.hook_post'][:, :, neurons].mean((0, 1))
        return value
    return [(f'blocks.{layer}.mlp.hook_post', ablate_neurons_hook)]


def save_activation(value, hook):
        """
        Add a hook using this method and an activation label then use the label to access the activation:
        act = model.hook_dict[act_label].ctx['activation'][:, :, 669]
        """
        hook.ctx['activation'] = value #.detach().cpu().to(torch.float16)
        return value