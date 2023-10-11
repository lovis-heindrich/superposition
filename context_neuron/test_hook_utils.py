import torch
import hook_utils


# Fixtures
class DummyHook:
    def __init__(self):
        self.ctx = {}


# Tests
def test_get_ablate_neuron_hook__single_neuron():
    act_name, hook = hook_utils.get_ablate_neuron_hook(3, 0, -0.2, 'post')
    
    assert act_name == "blocks.3.mlp.hook_post"
    # example activation tensor with 1 batch, 1 pos, and 3 neurons
    torch.testing.assert_close(hook(torch.tensor([[[0.0, 1.0, 2.0]]]), None), torch.tensor([[[-0.2, 1.0, 2.0]]]))


def test_get_ablate_neuron_hook__multiple_neurons():
    act_name, hook = hook_utils.get_ablate_neuron_hook(3, torch.tensor([0, 1]), torch.tensor([-0.2, -0.4]), 'post')
    
    assert act_name == "blocks.3.mlp.hook_post"
    # example activation tensor with 1 batch, 1 pos, and 3 neurons
    torch.testing.assert_close(hook(torch.tensor([[[0.0, 1.0, 2.0]]]), None), torch.tensor([[[-0.2, -0.4, 2.0]]]))


def test_get_ablate_context_neurons_hooks():
    hooks = hook_utils.get_ablate_context_neurons_hooks([(2, 0), (3, 1)], [-0.2, -0.4])
    
    first_act_name, first_hook = hooks[0]
    second_act_name, second_hook = hooks[1]

    assert first_act_name == "blocks.2.mlp.hook_post"
    assert second_act_name == "blocks.3.mlp.hook_post"

    # example activation tensor with 1 batch, 1 pos, and 3 neurons
    torch.testing.assert_close(first_hook(torch.tensor([[[0.0, 1.0, 2.0]]]), None), torch.tensor([[[-0.2, 1.0, 2.0]]]))
    torch.testing.assert_close(second_hook(torch.tensor([[[0.0, 1.0, 2.0]]]), None), torch.tensor([[[0.0, -0.4, 2.0]]]))


def test_save_activation_does_not_modify_acts():
    torch.testing.assert_close(hook_utils.save_activation(torch.tensor([0.0]), DummyHook()), torch.tensor([0.0]))


def test_save_activation():
    dummy_hook = DummyHook()
    hook_utils.save_activation(torch.tensor([0.0]), dummy_hook)

    assert dummy_hook.ctx['activation'] == torch.tensor([0.0])