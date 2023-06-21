import torch
from haystack_utils import get_average_loss, load_txt_data
from transformer_lens import HookedTransformer


def test_get_average_loss_unbatched():
    model = HookedTransformer.from_pretrained("pythia-70m-v0", fold_ln=True, device="cuda")

    kde_french = load_txt_data("kde4_french.txt")
    prompt = kde_french[0:1]
    tokens = model.to_tokens(prompt)

    loss = model(tokens, return_type="loss", loss_per_token=True)
    avg_loss = get_average_loss(prompt, model, batch_size=1, crop_context=-1, fwd_hooks=[], positionwise=False)
    
    torch.testing.assert_close(loss.mean(), avg_loss)


def test_get_average_loss_batched():
    model = HookedTransformer.from_pretrained("pythia-70m-v0", fold_ln=True, device="cuda")

    kde_french = load_txt_data("kde4_french.txt")
    prompt = kde_french[:5]

    # ensure prompts are equal length
    tokens = model.to_tokens(prompt)[:, :10]

    loss = model(tokens, return_type="loss", loss_per_token=True)
    print(loss.shape)
    avg_loss = get_average_loss(prompt, model, batch_size=1, crop_context=-1, fwd_hooks=[], positionwise=False)

    torch.testing.assert_close(loss.mean(), avg_loss)
