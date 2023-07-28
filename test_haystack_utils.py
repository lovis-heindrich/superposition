import torch
import haystack_utils
from transformer_lens import HookedTransformer


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
    print(prompts)

    loss = model(prompts, return_type="loss", loss_per_token=True)
    avg_loss = haystack_utils.get_average_loss(prompts, model, crop_context=-1, fwd_hooks=[], positionwise=False)

    torch.testing.assert_close(loss.mean(), avg_loss)


def test_weighted_mean():
    mean_acts = [torch.tensor([0.0]), torch.tensor([1.0])]
    weighted_mean = haystack_utils.weighted_mean(mean_acts, [1, 2])
    
    torch.testing.assert_close(weighted_mean, torch.tensor([2/3]))


def test_DLA():
    model = HookedTransformer.from_pretrained("pythia-70m-v0", fold_ln=True, device="cuda")
    test_prompts = ["chicken"]
    logit_attributions, labels = haystack_utils.DLA(test_prompts, model)

    assert logit_attributions.shape[0] == 1
    assert logit_attributions.shape[1] == 7
    

def test_top_k_with_exclude():
    numbers = torch.tensor([0, 1, 2, 3])
    
    assert torch.topk(numbers, 1) == haystack_utils.top_k_with_exclude(numbers, 1, [0])
    assert haystack_utils.top_k_with_exclude(numbers, 2, numbers) == torch.tensor([])