# %%
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer, ActivationCache, utils, patching
from jaxtyping import Float, Int, Bool
from torch import Tensor
from tqdm.auto import tqdm
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display, clear_output
from einops import einsum
import plotly.express as px
import numpy as np
import pandas as pd
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)

import haystack_utils

# %%
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device)

# %%
activate_neurons_fwd_hooks, deactivate_neurons_fwd_hooks = haystack_utils.get_context_ablation_hooks(3, [669], model)
all_ignore, _ = haystack_utils.get_weird_tokens(model, plot_norms=False)
# %%
german_data = haystack_utils.load_json_data("data/german_europarl.json")[:200]
common_tokens = haystack_utils.get_common_tokens(german_data, model, all_ignore, k=50)

# %%
options = [" Vorschlägen", " häufig", " schließt", " beweglich"]
all_prompts = {}
for option in options:
    all_prompts[option]=haystack_utils.generate_random_prompts(option, model, common_tokens, 200, length=20)

# %% 
random_prompts = haystack_utils.generate_random_prompts(" Vorschlägen", model, common_tokens, 100, length=20)[:, :-4]
loss, cache = model.run_with_cache(random_prompts)
mean_attention_activations = [] 
for layer in range(6):
    activation = cache[f'blocks.{layer}.attn.hook_z'].mean((0, 1))
    mean_attention_activations.append(activation)
mean_attention_activations = torch.stack(mean_attention_activations)

# %%
head_ablation_losses = {}
names = ["Original"] + [f"L{layer}H{head}" for layer in range(5) for head in range(8)]

for option in tqdm(options):
    head_ablation_losses[option] = {}
    prompts = all_prompts[option]
    str_tokens = model.to_str_tokens(model.to_tokens(option, prepend_bos=False))
    for pos in range(1-len(str_tokens), -1, 1):
        losses = []
        ablated_pos = str_tokens[pos]
        for layer in range(6):
            for head in range(8):
                ablate_head_hook = [haystack_utils.get_ablate_attention_hook(layer, head, mean_attention_activations, pos=pos)]
                with model.hooks(fwd_hooks=ablate_head_hook):
                    loss = model(prompts, return_type="loss", loss_per_token=True)[:, -1].tolist()
                    losses.append(loss)

        original_loss = model(prompts, return_type="loss", loss_per_token=True)[:, -1].tolist()
        all_losses = [original_loss] + losses
        head_ablation_losses[option][ablated_pos] = all_losses

# %%
with open('data/bigram_attention/head_ablation_losses.json', 'w') as f:
    json.dump(head_ablation_losses, f)

print(head_ablation_losses.keys())
print(head_ablation_losses[" Vorschlägen"].keys())
# %%
option = " Vorschlägen"
token = "lä"#"lä"
haystack_utils.plot_barplot(head_ablation_losses[option][token], names, legend=False, title=f"Loss for ablated attention heads on '{token}' of '{option}'")
# %%
