import torch
from haystack_utils import get_average_loss, load_txt_data
from transformer_lens import HookedTransformer


def test_get_average_loss():
    model = HookedTransformer.from_pretrained("pythia-70m-v0", fold_ln=True, device="cuda")

    kde_french = load_txt_data("kde4_french.txt")
    prompt = kde_french[0:1]
    tokens = model.to_tokens(prompt)

    loss = model(tokens, return_type="loss", loss_per_token=True)
    avg_loss = get_average_loss(prompt, model, batch_size=1, crop_context=-1, fwd_hooks=[], positionwise=False)
    
    torch.testing.assert_close(loss.mean(), avg_loss)


