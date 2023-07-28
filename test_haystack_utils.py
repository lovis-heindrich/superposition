import torch
import haystack_utils
from transformer_lens import HookedTransformer

import hook_utils


def test_get_average_loss_unbatched():
    model = HookedTransformer.from_pretrained("pythia-70m-v0", fold_ln=True, device="cuda")

    kde_french = haystack_utils.load_txt_data("data/kde4_french.txt")
    prompt = kde_french[0:1]
    tokens = model.to_tokens(prompt)

    loss = model(tokens, return_type="loss", loss_per_token=True)
    avg_loss = haystack_utils.get_average_loss(prompt, model, crop_context=-1, fwd_hooks=[], positionwise=False)
    
    torch.testing.assert_close(loss.mean(), avg_loss)


def test_get_average_loss_batched():
    model = HookedTransformer.from_pretrained("pythia-70m-v0", fold_ln=True, device="cuda")
    kde_french = haystack_utils.load_txt_data("data/kde4_french.txt")
    # token rows are of equal length and prepended with BOS
    prompts = kde_french[:5]
    tokens = model.to_tokens(prompts)[:, :10]
    prompts = model.to_string(tokens[:, 1:])  # remove BOS

    loss = model(prompts, return_type="loss", loss_per_token=True)
    avg_loss = haystack_utils.get_average_loss(prompts, model, crop_context=-1, fwd_hooks=[], positionwise=False)

    torch.testing.assert_close(loss.mean(), avg_loss)


def test_weighted_mean():
    mean_acts = [torch.tensor([0.0]), torch.tensor([1.0])]
    weighted_mean = haystack_utils.weighted_mean(mean_acts, [1, 2])
    
    torch.testing.assert_close(weighted_mean, torch.tensor([2/3]))


def test_get_mlp_activations_with_mean():
    model = HookedTransformer.from_pretrained("pythia-70m", fold_ln=True, device="cuda")
    german_data = haystack_utils.load_json_data("data/german_europarl.json")[:200]
    acts = haystack_utils.get_mlp_activations(german_data, 3, model, mean=True)
    
    torch.testing.assert_close(acts[669].item(), 3.47, atol=0.1, rtol=0.0)


def test_get_mlp_activations_with_mean_neurons():
    model = HookedTransformer.from_pretrained("pythia-70m", fold_ln=True, device="cuda")
    german_data = haystack_utils.load_json_data("data/german_europarl.json")[:200]
    acts = haystack_utils.get_mlp_activations(german_data, 3, model, mean=True, neurons=torch.tensor([669]))
    
    torch.testing.assert_close(acts, torch.tensor([3.47]).cuda(), atol=0.1, rtol=0.0)


def test_DLA():
    model = HookedTransformer.from_pretrained("pythia-70m", fold_ln=True, device="cuda")
    test_prompts = ["chicken"]
    logit_attributions, labels = haystack_utils.DLA(test_prompts, model)

    assert logit_attributions.shape[0] == 1
    assert logit_attributions.shape[1] == 7
    

def test_top_k_with_exclude():
    numbers = torch.tensor([0.0, 1.0, 2.0, 3.0])
    values, indices = haystack_utils.top_k_with_exclude(numbers, 2, torch.tensor([2]))
    
    torch.testing.assert_close(values, torch.tensor([3.0, 1.0]))


def test_get_direct_effect_does_nothing_without_hooks():
    test_prompt = "chicken"
    model = HookedTransformer.from_pretrained("pythia-70m-v0", fold_ln=True, device="cuda")

    original_loss, ablated_loss, direct_and_activated_loss, activated_loss = haystack_utils.get_direct_effect(test_prompt, model, [], [],
                                    deactivated_components=("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out"),
                                    activated_components=("blocks.5.hook_mlp_out",), return_type = 'loss')

    assert isinstance(original_loss, float)
    assert all(loss == original_loss for loss in [ablated_loss, direct_and_activated_loss, activated_loss])
    

def test_get_direct_effect_batched():
    test_prompts = ["chicken", "chicken"]
    model = HookedTransformer.from_pretrained("pythia-70m-v0", fold_ln=True, device="cuda")

    original_loss, ablated_loss, direct_and_activated_loss, activated_loss = haystack_utils.get_direct_effect(test_prompts, model, [], [],
                                    deactivated_components=("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out"),
                                    activated_components=("blocks.5.hook_mlp_out",), return_type = 'loss')

    assert isinstance(original_loss, torch.Tensor)
    assert all(loss[0] == original_loss[0] for loss in [ablated_loss, direct_and_activated_loss, activated_loss])

    
def test_get_direct_effect_hooked():
    test_prompt = "chicken"
    hook = hook_utils.get_ablate_neuron_hook(3, 669, -0.2, 'post')
    model = HookedTransformer.from_pretrained("pythia-70m", fold_ln=True, device="cuda")

    original_loss, ablated_loss, direct_and_activated_loss, activated_loss = haystack_utils.get_direct_effect(test_prompt, model, [hook], [],
                                    deactivated_components=("blocks.4.hook_attn_out", "blocks.5.hook_attn_out", "blocks.4.hook_mlp_out"),
                                    activated_components=("blocks.5.hook_mlp_out",), return_type = 'loss')

    assert isinstance(original_loss, float)
    assert all(loss != original_loss for loss in [ablated_loss, direct_and_activated_loss, activated_loss])


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
