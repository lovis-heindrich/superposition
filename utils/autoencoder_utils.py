import os
import sys
sys.path.append('../')  # Add the parent directory to the system path

import torch.nn.functional as F
from jaxtyping import Int, Float
from torch import Tensor
import einops
from dataclasses import dataclass
from typing import Literal
from transformer_lens import HookedTransformer
import torch
from collections import Counter
import logging
from tqdm import tqdm

import utils.haystack_utils as haystack_utils
from sparse_coding.train_autoencoder import AutoEncoder


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


@dataclass
class AutoEncoderConfig:
    layer: int
    act_name: Literal["mlp.hook_post", "hook_mlp_out"]
    expansion_factor: int
    l1_coeff: float

    @property
    def encoder_hook_point(self) -> str:
        return f"blocks.{self.layer}.{self.act_name}"


def batch_prompts(data: list[str], model: HookedTransformer, seq_len: int):
    tensors = []
    for prompt in tqdm(data):
        tokens = model.to_tokens(prompt).cpu()
        tensors.append(tokens)

    batched_tensor = torch.cat(tensors, dim=1)
    batched_tensor = einops.rearrange(
        batched_tensor[
            :, : (batched_tensor.shape[1] - batched_tensor.shape[1] % seq_len)
        ],
        "batch (x seq_len) -> (batch x) seq_len",
        seq_len=seq_len,
    )

    batched_bos = torch.zeros(batched_tensor.shape[0], 1, dtype=torch.long)
    batched_bos[:] = model.tokenizer.bos_token_id
    batched_tensor = torch.cat([batched_bos, batched_tensor], dim=1)
    batched_tensor = batched_tensor[torch.randperm(batched_tensor.shape[0])]
    batched_tensor = batched_tensor.to(torch.int32)
    return batched_tensor


def custom_forward(
    enc: AutoEncoder, x: Float[Tensor, "batch d_in"], neuron: int, activation: float
):
    x_cent = x - enc.b_dec
    acts = F.relu(x_cent @ enc.W_enc + enc.b_enc)
    acts[:, neuron] = activation
    x_reconstruct = acts @ enc.W_dec + enc.b_dec
    l2_loss = (x_reconstruct - x).pow(2).sum(-1).mean(0)
    l1_loss = enc.l1_coeff * (acts.abs().sum())
    loss = l2_loss + l1_loss
    return loss, x_reconstruct, acts, l2_loss, l1_loss


def encoder_dla_batched(
    tokens: Int[Tensor, "batch pos"],
    model: HookedTransformer,
    encoder: AutoEncoder,
    cfg: AutoEncoderConfig,
) -> Float[Tensor, "batch pos n_neurons"]:
    encoder_hook_point = f"blocks.{cfg.layer}.{cfg.act_name}"

    batch_dim, seq_len = tokens.shape
    _, cache = model.run_with_cache(tokens)
    mlp_activations = cache[encoder_hook_point][:, :-1]
    _, _, mid_acts, _, _ = encoder(mlp_activations)

    W_U_token = einops.rearrange(
        model.W_U[:, tokens.flatten()],
        "d_res (batch pos) -> d_res batch pos",
        batch=batch_dim,
        pos=seq_len,
    )
    if cfg.act_name == "mlp.hook_post":
        W_out_U_token = einops.einsum(
            model.W_out[cfg.layer],
            W_U_token,
            "d_mlp d_res, d_res batch pos -> d_mlp batch pos",
        )
        W_dec_W_out_U_token = einops.einsum(
            encoder.W_dec,
            W_out_U_token,
            "d_dec d_mlp, d_mlp batch pos -> d_dec batch pos",
        )
        dla = einops.einsum(
            mid_acts,
            W_dec_W_out_U_token[:, :, 1:],
            "batch pos d_dec, d_dec batch pos -> batch pos d_dec",
        )
    elif cfg.act_name == "hook_mlp_out":
        W_dec_U_token = einops.einsum(
            encoder.W_dec, W_U_token, "d_dec d_res, d_res batch pos -> d_dec batch pos"
        )
        dla = einops.einsum(
            mid_acts,
            W_dec_U_token[:, :, 1:],
            "batch pos d_dec, d_dec batch pos -> batch pos d_dec",
        )
    else:
        raise ValueError("Unknown act_name")
    scale = cache["ln_final.hook_scale"][:, :-1]
    dla = dla / scale
    return dla


def get_trigram_token_dla(
    model: HookedTransformer,
    encoder: AutoEncoder,
    encoder_neuron: int,
    trigram: str,
    cfg: AutoEncoderConfig,
):
    last_trigram_token = model.to_str_tokens(model.to_tokens(trigram))[-1]
    correct_token = model.to_single_token(last_trigram_token)

    if cfg.act_name == "mlp.hook_post":
        boosts = (encoder.W_dec[encoder_neuron] @ model.W_out[cfg.layer]) @ model.W_U
    else:
        boosts = encoder.W_dec[encoder_neuron] @ model.W_U
    trigram_token_dla = boosts[correct_token].item()
    logging.info(f"'{last_trigram_token}' DLA = {trigram_token_dla:.2f}")
    return trigram_token_dla


def get_directions_from_dla(
    dla: Float[Tensor, "n_neurons"], cutoff_dla=0.2, max_directions=3
):
    top_dla, top_neurons = torch.topk(dla, max_directions, largest=True)
    directions = []
    for i in range(max_directions):
        if top_dla[i].item() > cutoff_dla:
            directions.append(top_neurons[i].item())
    logging.info(
        f"Top {max_directions} directions with DLA > {cutoff_dla}: {directions}"
    )
    return directions


def get_trigram_dataset_examples(
    model: HookedTransformer,
    trigram: str,
    german_data: list[str],
    prompt_length=50,
    max_prompts=100,
):
    # Boosted tokens with reasonable logprob
    middle_trigram_token = model.to_str_tokens(model.to_tokens(trigram))[-2]
    token = model.to_single_token(middle_trigram_token)
    token_prompts = []
    occurrences = []
    for prompt in german_data:
        tokenized_prompt = model.to_tokens(prompt).flatten()
        token_indices = torch.where(tokenized_prompt == token)
        if len(token_indices[0]) > 0:
            for i in token_indices[0]:
                if i > prompt_length:
                    new_prompt = tokenized_prompt[i - prompt_length : i + 1]
                    occurrences.append(
                        "".join(model.to_str_tokens(tokenized_prompt[i - 1 : i + 2]))
                    )
                    token_prompts.append(new_prompt)
        if len(token_prompts) >= max_prompts:
            break
    token_prompts = torch.stack(token_prompts)
    logging.info(
        f"Found {token_prompts.shape[0]} prompts with token '{middle_trigram_token}'"
    )
    logging.info(Counter(occurrences))
    return token_prompts


def get_zero_ablate_encoder_direction_hook(
    encoder: AutoEncoder, encoder_neuron, cfg: AutoEncoderConfig
):
    def zero_feature_hook(value, hook):
        _, x_reconstruct, _, _, _ = custom_forward(
            encoder, value[:, -1], encoder_neuron, 0
        )
        value[:, -1] = x_reconstruct
        return value

    return [(cfg.encoder_hook_point, zero_feature_hook)]


def get_encode_activations_hook(
    encoder: AutoEncoder, encoder_neuron, cfg: AutoEncoderConfig
):
    def encode_activations_hook(value, hook):
        _, x_reconstruct, acts, _, _ = encoder(value[:, -1])
        feature_activations = acts[:, encoder_neuron]
        value[:, -1] = x_reconstruct
        return value

    return [(cfg.encoder_hook_point, encode_activations_hook)]


def get_direction_logit_and_logprob_boost(
    trigram_tokens: Int[Tensor, "batch pos"],
    encoder: AutoEncoder,
    encoder_neuron,
    model: HookedTransformer,
    trigram: str,
    common_tokens: Int[Tensor, "tokens"],
    all_ignore: Int[Tensor, "tokens"],
    cfg: AutoEncoderConfig,
):
    last_trigram_token = model.to_str_tokens(model.to_tokens(trigram))[-1]

    zero_direction_hook = get_zero_ablate_encoder_direction_hook(
        encoder, encoder_neuron, cfg
    )
    encode_activations_hook = get_encode_activations_hook(encoder, encoder_neuron, cfg)

    with model.hooks(encode_activations_hook):
        logits_active = model(trigram_tokens[:, :-1], return_type="logits")[:, -1]

    with model.hooks(zero_direction_hook):
        logits_inactive = model(trigram_tokens[:, :-1], return_type="logits")[:, -1]

    last_trigram_token_tokenized = model.to_single_token(last_trigram_token)
    last_token_logit_encoded = (
        logits_active[:, last_trigram_token_tokenized].mean(0).item()
    )
    last_token_logit_zeroed = (
        logits_inactive[:, last_trigram_token_tokenized].mean(0).item()
    )
    logging.info(
        f"Logit '{last_trigram_token}' when reconstructing activations with encoder: {last_token_logit_encoded:.2f}"
    )
    logging.info(
        f"Logit '{last_trigram_token}' when reconstructing activations with encoder zeroing out the direction: {last_token_logit_zeroed:.2f}"
    )

    logprobs_active = logits_active.log_softmax(dim=-1)
    logprobs_inactive = logits_inactive.log_softmax(dim=-1)
    last_token_logprob_encoded = (
        logprobs_active[:, last_trigram_token_tokenized].mean(0).item()
    )
    last_token_logprob_zeroed = (
        logprobs_inactive[:, last_trigram_token_tokenized].mean(0).item()
    )
    logging.info(
        f"Logprob '{last_trigram_token}' when reconstructing activations with encoder: {last_token_logprob_encoded:.2f}"
    )
    logging.info(
        f"Logprob '{last_trigram_token}' when reconstructing activations with encoder zeroing out the direction: {last_token_logprob_zeroed:.2f}"
    )

    boosts = (logprobs_active - logprobs_inactive).mean(0)
    boosts[logprobs_active.mean(0) < -7] = 0
    boosts[all_ignore] = 0
    top_boosts, top_tokens = torch.topk(boosts, 15)
    non_zero_boosts = top_boosts != 0
    top_deboosts, top_deboosted_tokens = torch.topk(boosts, 15, largest=False)
    non_zero_deboosts = top_deboosts != 0
    boosted_tokens = (
        model.to_str_tokens(top_tokens[non_zero_boosts]),
        top_boosts[non_zero_boosts].tolist(),
    )
    deboosted_tokens = (
        model.to_str_tokens(top_deboosted_tokens[non_zero_deboosts]),
        top_deboosts[non_zero_deboosts].tolist(),
    )
    logging.info(f"Top boosted: {boosted_tokens}")
    logging.info(f"Top deboosted: {deboosted_tokens}")

    return (
        last_token_logit_encoded,
        last_token_logit_zeroed,
        last_token_logprob_encoded,
        last_token_logprob_zeroed,
        boosted_tokens,
        deboosted_tokens,
    )


def print_direction_activations(
    data: list[str],
    model: HookedTransformer,
    encoder: AutoEncoder,
    encoder_neuron: int,
    cfg: AutoEncoderConfig,
    mean_feature_activation: float = 5,
    threshold=0,
):
    for prompt in data:
        _, cache = model.run_with_cache(prompt, names_filter=cfg.encoder_hook_point)
        acts = cache[cfg.encoder_hook_point].squeeze(0)
        _, _, mid_acts, _, _ = encoder(acts)
        neuron_act = mid_acts[:, encoder_neuron]
        if neuron_act.max() > threshold:
            str_tokens = model.to_str_tokens(model.to_tokens(prompt))
            haystack_utils.clean_print_strings_as_html(
                str_tokens, neuron_act.cpu().numpy(), max_value=mean_feature_activation
            )


def get_context_effect_on_feature_activations(
    model: HookedTransformer,
    tokens: Int[Tensor, "batch pos"],
    encoder: AutoEncoder,
    encoder_neuron: int,
    deactivate_context_hook,
    cfg: AutoEncoderConfig,
):
    context_active_loss = (
        model(tokens, return_type="loss", loss_per_token=True)[:, -1].mean().item()
    )
    _, cache = model.run_with_cache(tokens, names_filter=cfg.encoder_hook_point)
    acts = cache[cfg.encoder_hook_point]
    _, _, mid_acts, _, _ = encoder(acts)
    feature_activation_context_active = mid_acts[:, -2, encoder_neuron].mean().item()

    with model.hooks(deactivate_context_hook):
        context_ablated_loss = (
            model(tokens, return_type="loss", loss_per_token=True)[:, -1].mean().item()
        )

        _, cache = model.run_with_cache(tokens, names_filter=cfg.encoder_hook_point)
        acts = cache[cfg.encoder_hook_point]
        _, _, mid_acts, _, _ = encoder(acts)
        feature_activation_context_inactive = (
            mid_acts[:, -2, encoder_neuron].mean().item()
        )

    logging.info(f"Mean loss context active: {context_active_loss:.2f}")
    logging.info(f"Mean loss context inactive: {context_ablated_loss:.2f}")

    logging.info(
        f"Mean feature activation when context neuron active: {feature_activation_context_active:.2f}"
    )
    logging.info(
        f"Mean feature activation with context neuron inactive: {feature_activation_context_inactive:.2f}"
    )
    return (
        context_active_loss,
        context_ablated_loss,
        feature_activation_context_active,
        feature_activation_context_inactive,
    )


def get_encoder_token_reconstruction_losses(
    tokens: Int[Tensor, "batch pos"],
    model: HookedTransformer,
    encoder: AutoEncoder,
    deactivate_context_hook,
    cfg: AutoEncoderConfig,
):
    # Set all features to context active or inactive
    _, cache = model.run_with_cache(tokens, names_filter=cfg.encoder_hook_point)
    acts_active = cache[cfg.encoder_hook_point][:, -2]

    with model.hooks(deactivate_context_hook):
        _, cache = model.run_with_cache(tokens, names_filter=cfg.encoder_hook_point)
        acts_inactive = cache[cfg.encoder_hook_point][:, -2]

    def activate_feature_hook(value, hook):
        _, x_reconstruct, _, _, _ = encoder(acts_active)
        value[:, -2] = x_reconstruct
        return value

    def deactivate_feature_hook(value, hook):
        _, x_reconstruct, _, _, _ = encoder(acts_inactive)
        value[:, -2] = x_reconstruct
        return value

    activate_hooks = [(cfg.encoder_hook_point, activate_feature_hook)]
    deactivate_hooks = [(cfg.encoder_hook_point, deactivate_feature_hook)]

    with model.hooks(activate_hooks):
        encoder_context_active_loss = model(
            tokens, return_type="loss", loss_per_token=True
        )[:, -1].mean()
        logging.info(
            f"Model loss when patching through encoder with context neuron active: {encoder_context_active_loss.item():.2f}"
        )

    with model.hooks(deactivate_hooks):
        encoder_context_inactive_loss = model(
            tokens, return_type="loss", loss_per_token=True
        )[:, -1].mean()
        logging.info(
            f"Model loss when patching through encoder with context neuron inactive: {encoder_context_inactive_loss.item():.2f}"
        )

    return encoder_context_active_loss, encoder_context_inactive_loss


def get_encoder_feature_reconstruction_losses(
    tokens: Int[Tensor, "batch pos"],
    encoder: AutoEncoder,
    model: HookedTransformer,
    encoder_neuron: int,
    feature_activation_context_active: float,
    feature_activation_context_inactive: float,
    cfg: AutoEncoderConfig,
):
    # Set single feature to context active or inactive
    def activate_feature_hook(value, hook):
        _, x_reconstruct, _, _, _ = custom_forward(
            encoder, value[:, -2], encoder_neuron, feature_activation_context_active
        )
        value[:, -2] = x_reconstruct
        return value

    def deactivate_feature_hook(value, hook):
        _, x_reconstruct, _, _, _ = custom_forward(
            encoder, value[:, -2], encoder_neuron, feature_activation_context_inactive
        )
        value[:, -2] = x_reconstruct
        return value

    def zero_feature_hook(value, hook):
        _, x_reconstruct, _, _, _ = custom_forward(
            encoder, value[:, -2], encoder_neuron, 0
        )
        value[:, -2] = x_reconstruct
        return value

    activate_hooks = [(cfg.encoder_hook_point, activate_feature_hook)]
    deactivate_hooks = [(cfg.encoder_hook_point, deactivate_feature_hook)]
    zero_ablate_hooks = [(cfg.encoder_hook_point, zero_feature_hook)]

    with model.hooks(activate_hooks):
        loss_encoder_direction_active = model(
            tokens, return_type="loss", loss_per_token=True
        )[:, -1].mean()
        logging.info(
            f"Mean loss when patching second trigram token through encoder: {loss_encoder_direction_active.item():.2f}"
        )

    with model.hooks(deactivate_hooks):
        loss_encoder_direction_inactive = model(
            tokens, return_type="loss", loss_per_token=True
        )[:, -1].mean()
        logging.info(
            f"Mean loss when patching second trigram token through encoder and setting N{encoder_neuron} to activation with context neuron inactive: {loss_encoder_direction_inactive.item():.2f}"
        )

    with model.hooks(zero_ablate_hooks):
        loss_encoder_direction_zeroed = model(
            tokens, return_type="loss", loss_per_token=True
        )[:, -1].mean()
        logging.info(
            f"Mean loss when patching second trigram token through encoder and setting N{encoder_neuron} to zero: {loss_encoder_direction_zeroed.item():.2f}"
        )

    return (
        loss_encoder_direction_active,
        loss_encoder_direction_inactive,
        loss_encoder_direction_zeroed,
    )
