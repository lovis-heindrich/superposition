import os
import sys
sys.path.append('../')  # Add the parent directory to the system path

import torch.nn.functional as F
from jaxtyping import Int, Float, Bool
from torch import Tensor
import einops
from dataclasses import dataclass
from typing import Literal
from transformer_lens import HookedTransformer
import torch
from collections import Counter
import logging
from tqdm import tqdm
import numpy as np
import json
import pickle
import plotly.express as px
import pandas as pd

import utils.haystack_utils as haystack_utils
from sparse_coding.autoencoder import AutoEncoder


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
    act_name: Literal["mlp.hook_post", "hook_mlp_out", "attn.hook_result"]
    expansion_factor: int
    l1_coeff: float
    d_in: int
    run_name: str | None = None
    reg: str = "l1",
    head_idx: int | None = None,

    @property
    def encoder_hook_point(self) -> str:
        return f"blocks.{self.layer}.{self.act_name}"
    
def get_component_baseline_losses(prompts, model, hook_name: str, head_idx: int | None=None, disable_tqdm=False, prepend_bos=False):
    # all_acts = []
    # for prompt in prompts:
    #     _, cache = model.run_with_cache(prompt, names_filter=hook_name, prepend_bos=prepend_bos)
        
    #     acts = cache[hook_name].squeeze(0)
    #     if head_idx is not None:
    #         acts = acts[:, head_idx]
    #     all_acts.append(acts.mean(0))
    # mean_act = torch.stack(all_acts).mean(0)

    # def mean_ablation_hook(value, hook):
    #     if head_idx is not None:
    #         value[:, :, head_idx] = mean_act
    #     else:
    #         value[:, :] = mean_act
    #     return value
    
    # mean_ablate_hook = [(hook_name, mean_ablation_hook)]
    # mean_abl_losses = []

    def zero_ablation_hook(value, hook):
        if head_idx is not None:
            value[:, :, head_idx] = 0
        else:    
            value[:, :] = 0
        return value

    zero_ablate_hook = [(hook_name, zero_ablation_hook)]

    losses = []
    zero_abl_losses = []
    for prompt in tqdm(prompts, disable=disable_tqdm):
        if isinstance(prompt, torch.Tensor):
            tokens = prompt
        else:
            tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
        loss = model(tokens, return_type="loss", loss_per_token=False).item()
        with model.hooks(fwd_hooks=zero_ablate_hook):
            zero_abl_loss = model(tokens, return_type="loss", loss_per_token=False).item()
        # with model.hooks(fwd_hooks=mean_ablate_hook):
        #     mean_abl_loss = model(tokens, return_type="loss", loss_per_token=False).item()
        # mean_abl_losses.append(mean_abl_loss)
        losses.append(loss)
        zero_abl_losses.append(zero_abl_loss)
    
    return np.mean(losses), np.mean(zero_abl_losses)
    
def get_loss_recovered(prompts, model, encoder, cfg: AutoEncoderConfig, disable_tqdm=False, prepend_bos=False):
    hook_name = f"blocks.{cfg.layer}.{cfg.act_name}"

    def zero_ablation_hook(value, hook):
        if cfg.head_idx:
            value[:, :, cfg.head_idx] = 0
        else:
            value[:, :] = 0
        return value

    def encode_activations_hook(value, hook):
        which = np.s_[0, :, cfg.head_idx] if cfg.head_idx else np.s_[0]
        _, x_reconstruct, _, _, _ = encoder(value[which])
        value[which] = x_reconstruct
        return value

    zero_ablate_hook = [(hook_name, zero_ablation_hook)]
    encode_mlp_hook = [(hook_name, encode_activations_hook)]

    losses = []
    zero_abl_losses = []
    recons_losses = []

    for prompt in tqdm(prompts, disable=disable_tqdm):
        tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
        loss = model(tokens, return_type="loss", loss_per_token=False).item()
        with model.hooks(fwd_hooks=zero_ablate_hook):
            zero_abl_loss = model(tokens, return_type="loss", loss_per_token=False).item()
        with model.hooks(fwd_hooks=encode_mlp_hook):
            recons_loss = model(tokens, return_type="loss", loss_per_token=False).item()
        losses.append(loss)
        zero_abl_losses.append(zero_abl_loss)
        recons_losses.append(recons_loss)

    loss_recovered = ((np.mean(zero_abl_losses) - np.mean(recons_losses))/(np.mean(zero_abl_losses) - np.mean(losses)))
    return loss_recovered

def get_l0(prompts, model, encoder, cfg:AutoEncoderConfig, disable_tqdm=False, prepend_bos=False):
    l0 = []
    for prompt in tqdm(prompts, disable=disable_tqdm):
        tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
        acts = get_acts(tokens, model, encoder, cfg)
        active_directions = (acts > 0).sum(1)
        l0.extend(active_directions.tolist())
    return np.mean(l0)

def get_feature_density(prompts, model, encoder, cfg, disable_tqdm=False, prepend_bos=False):
    feature_activations = torch.zeros(encoder.d_hidden).to(torch.long).cuda()
    total_num_tokens = 0
    for prompt in tqdm(prompts, disable=disable_tqdm):
        tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
        acts = get_acts(tokens, model, encoder, cfg)
        num_tokens = acts.shape[0]
        acts = (acts>0).sum(0)
        total_num_tokens += num_tokens
        feature_activations += acts
    feature_density = feature_activations / total_num_tokens
    return feature_density

def get_max_activations(prompts: list[str], model: HookedTransformer, encoder: AutoEncoder, cfg: AutoEncoderConfig):
    activations = []
    indices = []
    for prompt in tqdm(prompts):
        acts = get_acts(prompt, model, encoder, cfg)[:-1]
        value, index = acts.max(0)
        activations.append(value)
        indices.append(index)

    max_activation_per_prompt = torch.stack(activations)  # n_prompt x d_enc
    max_activation_token_index = torch.stack(indices)

    total_activations = max_activation_per_prompt.sum(0)
    print(f"Active directions on validation data: {total_activations.nonzero().shape[0]} out of {total_activations.shape[0]}")
    return max_activation_per_prompt, max_activation_token_index

def get_activations(encoder, cfg, encoder_name, prompts, model, save_path="/workspace", save_activations=True):
    path = f"{save_path}/data/{encoder_name}_activations.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            max_activations = data["max_activations"]
            max_activation_token_indices = data["max_activation_token_indices"]
    else:
        max_activations, max_activation_token_indices = get_max_activations(prompts, model, encoder, cfg)
        if save_activations:
            with open(path, "wb") as f:
                pickle.dump({"max_activations": max_activations, "max_activation_token_indices": max_activation_token_indices}, f)
    return max_activations, max_activation_token_indices

def get_direction_ablation_hook(encoder, direction, hook_pos=None):
    if type(direction) == int:
        direction = [direction]
    def subtract_direction_hook(value, hook):
        x_cent = value[0, :] - encoder.b_dec
        acts = F.relu(x_cent @ encoder.W_enc[:, direction] + encoder.b_enc[direction])
        direction_impact_on_reconstruction = einops.einsum(acts, encoder.W_dec[direction, :], "pos directions, directions d_mlp -> pos d_mlp") # + encoder.b_dec ???
        if hook_pos is not None:
            value[:, hook_pos, :] -= direction_impact_on_reconstruction[hook_pos]
        else:
            value[:, :] -= direction_impact_on_reconstruction
        return value
    return subtract_direction_hook

def evaluate_direction_ablation_single_prompt(prompt: str, encoder: AutoEncoder, model: HookedTransformer, direction: int | list[int], cfg: AutoEncoderConfig, pos: None | int = None, loss_per_token=False) -> float:
    """ Pos needs to be the absolute position of the token to ablate, negative indexing does not work """
    encoder_hook_point = f"blocks.{cfg.layer}.{cfg.act_name}"
    if pos is not None:
        original_loss = model(prompt, return_type="loss", loss_per_token=True)[0, pos].item()
    elif not loss_per_token:
        original_loss = model(prompt, return_type="loss").item()
    else:
        original_loss = model(prompt, return_type="loss", loss_per_token=True)
    
    with model.hooks(fwd_hooks=[(encoder_hook_point, get_direction_ablation_hook(encoder, direction, pos))]):
        if pos is not None:
            ablated_loss = model(prompt, return_type="loss", loss_per_token=True)[0, pos].item()
        elif not loss_per_token:
            ablated_loss = model(prompt, return_type="loss").item()
        else:
            ablated_loss = model(prompt, return_type="loss", loss_per_token=True)
    return original_loss, ablated_loss

def eval_ablation_token_rank(prompt: str, encoder: AutoEncoder, model: HookedTransformer, direction: int | list[int], cfg: AutoEncoderConfig, answer_token: str, pos: int = -2):
    encoder_hook_point = f"blocks.{cfg.layer}.{cfg.act_name}"
    answer_token_index = model.to_single_token(answer_token)

    logits = model(prompt, return_type="logits")[0, pos]
    answer_rank = (logits > logits[answer_token_index]).sum().item()
    answer_logprob = logits.log_softmax(dim=-1)[answer_token_index]
    answer_logit = logits[answer_token_index]

    with model.hooks(fwd_hooks=[(encoder_hook_point, get_direction_ablation_hook(encoder, direction, pos))]):
        ablated_logits = model(prompt, return_type="logits")[0, pos]
        ablated_answer_rank = (ablated_logits > ablated_logits[answer_token_index]).sum().item()
        ablated_answer_logprob = ablated_logits.log_softmax(dim=-1)[answer_token_index]
        ablated_answer_logit = ablated_logits[answer_token_index]
    return answer_logprob.item(), ablated_answer_logprob.item(), answer_logit.item(), ablated_answer_logit.item(), answer_rank, ablated_answer_rank

def eval_direction_tokens_global(max_activations, prompts, model, encoder, cfg, percentage_threshold = 0.25):
    max_activation_value = max_activations.max(dim=0)[0]
    threshold_per_direction = max_activation_value * percentage_threshold
    token_wise_activations = torch.zeros((encoder.d_hidden, model.cfg.d_vocab), dtype=torch.int32)

    for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        tokens = model.to_tokens(prompt).flatten().cpu()
        activations = get_acts(prompt, model, encoder, cfg).cpu()
        num_tokens = activations.shape[0]
        for position in range(10, num_tokens):
            valid_directions = torch.argwhere(activations[position] > threshold_per_direction).flatten().cpu()
            if len(valid_directions) > 0:
                token_wise_activations[valid_directions, tokens[position]] += 1
    return token_wise_activations

@torch.no_grad()
def get_acts(prompt: str | Tensor, model: HookedTransformer, encoder: AutoEncoder, cfg: AutoEncoderConfig):
    _, cache = model.run_with_cache(prompt, names_filter=cfg.encoder_hook_point)
    acts = cache[cfg.encoder_hook_point].squeeze(0)
    if cfg.act_name == 'attn.hook_result':
        acts = acts[:, cfg.head_idx, :]
    _, _, mid_acts, _, _ = encoder(acts)
    return mid_acts


def get_max_activations(prompts: list[str], model: HookedTransformer, encoder: AutoEncoder, cfg: AutoEncoderConfig):
    activations = []
    indices = []
    for prompt in tqdm(prompts):
        acts = get_acts(prompt, model, encoder, cfg)[:-1]
        value, index = acts.max(0)
        activations.append(value)
        indices.append(index)

    max_activation_per_prompt = torch.stack(activations)  # n_prompt x d_enc
    max_activation_token_index = torch.stack(indices)

    total_activations = max_activation_per_prompt.sum(0)
    print(f"Active directions on validation data: {total_activations.nonzero().shape[0]} out of {total_activations.shape[0]}")
    return max_activation_per_prompt, max_activation_token_index

def get_top_activating_examples_for_direction(prompts, direction, max_activations_per_prompt: Tensor, max_activation_token_indices, k=10, mode: Literal["lower", "middle", "upper", "top"]="top"):
    
    sorted_activations = max_activations_per_prompt[:, direction].sort().values
    num_non_zero_activations = sorted_activations[sorted_activations > 0].shape[0]

    max_activation = sorted_activations[-1]
    if mode=="upper":
        index = torch.argwhere(sorted_activations > ((max_activation // 3) * 2)).min()
    elif mode == "middle":
        index = torch.argwhere(sorted_activations > ((max_activation // 3))).min()
    else:
        index = torch.argwhere(sorted_activations > ((max_activation // 10))).min()
    negative_index = sorted_activations.shape[0] - index

    activations = max_activations_per_prompt[:, direction]
    _, prompt_indices = activations.topk(num_non_zero_activations+1)
    if mode=="top":
        prompt_indices = prompt_indices[:k]
    else:
        prompt_indices = prompt_indices[negative_index:negative_index+k]
    prompt_indices = prompt_indices[:num_non_zero_activations]

    top_prompts = [prompts[i] for i in prompt_indices]
    token_indices = max_activation_token_indices[prompt_indices, direction]
    return top_prompts, token_indices

def act_name_to_d_in(model: HookedTransformer, act_name: str):
    if act_name == "mlp.hook_post" or act_name == "mlp.hook_pre":
        return model.cfg.d_mlp
    elif act_name == "hook_mlp_out":
        return model.cfg.d_model
    elif act_name == "attn.hook_result":
        return model.cfg.d_model
    else:
        raise ValueError("Act name not recognised: ", act_name)


def load_encoder(save_name, model_name, model: HookedTransformer, save_path="."):
    path = f"{save_path}/{model_name}/{save_name}"
    with open(f"{path}.json", "r") as f:
        cfg = json.load(f)
        print(cfg)
    
    if "d_in" in cfg:
        d_in = cfg["d_in"]
    else:
        d_in = act_name_to_d_in(model, cfg['act'])

    if "reg" not in cfg:
        cfg["reg"] = "l1"
    
    cfg = AutoEncoderConfig(
        cfg["layer"], cfg["act"], cfg["expansion_factor"], cfg["l1_coeff"], d_in, reg=cfg["reg"], head_idx=cfg['head_idx'] if 'head_idx' in cfg else None
    )

    d_hidden = cfg.d_in * cfg.expansion_factor

    encoder = AutoEncoder(d_hidden, cfg.l1_coeff, cfg.d_in, reg=cfg.reg)
    encoder.load_state_dict(torch.load(f"{path}.pt"))
    encoder.to(get_device())
    return encoder, cfg

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

def get_encoder_feature_frequencies(data: list[str], model: HookedTransformer, encoder: AutoEncoder, cfg: AutoEncoderConfig):
    num_feature_activations = torch.zeros(encoder.d_hidden).to(get_device())
    mean_active = []
    total_tokens = 0
    for prompt in tqdm(data):
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(
            tokens, names_filter=f"blocks.{cfg.layer}.{cfg.act_name}"
            )
        acts = cache[f"blocks.{cfg.layer}.{cfg.act_name}"].squeeze(0)
        _, _, mid_acts, _, _ = encoder(acts)
        num_feature_activations = num_feature_activations + (mid_acts>0).sum(dim=0)
        active_features = (mid_acts > 0).sum(dim=1).float().mean(dim=0).item()
        mean_active.append(active_features)
        total_tokens += torch.numel(tokens)

    active_features = (num_feature_activations > 0).sum().item()
    feature_frequencies = num_feature_activations / total_tokens
    print(f"Number of active features over {total_tokens} tokens: {active_features}")
    print(f"Number of average active features per token: {np.mean(mean_active):.2f}")
    # fig = px.histogram(feature_frequencies.cpu().numpy(), histnorm='probability', log_y=True, title="Histogram of feature frequencies", nbins=40)
    # fig.update_layout(xaxis_title="Feature frequency", yaxis_title="Probability", showlegend=False, width=600)
    return feature_frequencies

@torch.no_grad()
def evaluate_autoencoder_reconstruction(autoencoder: AutoEncoder, encoded_hook_name: str, data: list[str], model: HookedTransformer, reconstruction_loss_only: bool = False, show_tqdm=True):
    def encode_activations_hook(value, hook):
        value = value.squeeze(0)
        _, x_reconstruct, _, _, _ = autoencoder(value)
        return x_reconstruct.unsqueeze(0)
    reconstruct_hooks = [(encoded_hook_name, encode_activations_hook)]

    def zero_ablate_hook(value, hook):
        value[:] = 0
        return value
    zero_ablate_hooks = [(encoded_hook_name, zero_ablate_hook)]
    
    original_losses = []
    reconstruct_losses = []
    zero_ablation_losses = []
    for prompt in tqdm(data, disable=(not show_tqdm)):
        with model.hooks(reconstruct_hooks):
            reconstruct_loss = model(prompt, return_type="loss")
        reconstruct_losses.append(reconstruct_loss.item())
        if not reconstruction_loss_only:
            original_loss = model(prompt, return_type="loss")
            with model.hooks(zero_ablate_hooks):
                zero_ablate_loss = model(prompt, return_type="loss")
            original_losses.append(original_loss.item())
            zero_ablation_losses.append(zero_ablate_loss.item())

    if reconstruction_loss_only:
        return np.mean(reconstruct_losses)
    logging.info(f"Average loss increase after encoding: {(np.mean(reconstruct_losses) - np.mean(original_losses)):.4f}")
    return np.mean(original_losses), np.mean(reconstruct_losses), np.mean(zero_ablation_losses)

@torch.no_grad()
def train_autoencoder_evaluate_autoencoder_reconstruction(autoencoder: AutoEncoder, cfg: AutoEncoderConfig, encoded_hook_name: str, data: list[str], model: HookedTransformer, reconstruction_loss_only: bool = False, show_tqdm=True, prepend_bos=True):
    '''Also grabs histogram of acts > 0'''
    act_nonzero_sums = torch.zeros(autoencoder.d_hidden).cuda()
    act_nonzero_counts = torch.zeros(autoencoder.d_hidden).cuda()

    def encode_activations_hook(value: Float[Tensor, "batch token [head] d_in"], hook):
        nonlocal act_nonzero_sums
        nonlocal act_nonzero_counts
        if len(value.shape) == 4:
            which = np.s_[0, :, cfg['head_idx']]
        else:
            which = np.s_[0]
        
        _, x_reconstruct, acts, _, _ = autoencoder(value[which])
        act_nonzero_sums += acts.sum(0)
        act_nonzero_counts += (acts > 0).sum(0)

        value[which] = x_reconstruct
        return value
    
    reconstruct_hooks = [(encoded_hook_name, encode_activations_hook)]

    def zero_ablate_hook(value, hook):
        if len(value.shape) == 4:
            which = np.s_[:, :, cfg['head_idx']]
        else:
            which = np.s_[:]

        value[which] = 0
        return value
    zero_ablate_hooks = [(encoded_hook_name, zero_ablate_hook)]
    
    original_losses = []
    reconstruct_losses = []
    zero_ablation_losses = []

    for prompt in tqdm(data, disable=(not show_tqdm)):
        if isinstance(data, torch.Tensor):
            tokens = prompt
        else:
            tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
        with model.hooks(reconstruct_hooks):
            reconstruct_loss = model(tokens, return_type="loss")
        reconstruct_losses.append(reconstruct_loss.item())
        if not reconstruction_loss_only:
            original_loss = model(tokens, return_type="loss")
            with model.hooks(zero_ablate_hooks):
                zero_ablate_loss = model(tokens, return_type="loss")
            original_losses.append(original_loss.item())
            zero_ablation_losses.append(zero_ablate_loss.item())

    act_nonzero_means = act_nonzero_sums / act_nonzero_counts

    fig = px.histogram(act_nonzero_means.cpu().detach(), title="Non-zero activation means over encoder features")
    fig.update_layout({
        "showlegend": False
    })
    return np.mean(reconstruct_losses), fig

@torch.no_grad()
def batched_reconstruction_loss(encoder: AutoEncoder, encoded_hook_name: str, data: list[str], model: HookedTransformer, batch_size: int):
    """ For some reason slower than non batched? """
    def encode_activations_hook(value, hook):
        current_batch_size = value.shape[0]
        value = einops.rearrange(value, "b s d -> (b s) d")
        _, x_reconstruct, _, _, _ = encoder(value)
        x_reconstruct = einops.rearrange(x_reconstruct, "(b s) d -> b s d", b=current_batch_size)
        return x_reconstruct
    reconstruct_hooks = [(encoded_hook_name, encode_activations_hook)]

    losses = []
    eos_token = model.tokenizer.eos_token_id
    # Generator
    batches = (data[i:i + batch_size] for i in range(0, len(data), batch_size))
    for batch in batches:
        with model.hooks(reconstruct_hooks):
            tokens = model.to_tokens(batch)
            reconstruct_loss = model(tokens, return_type="loss", loss_per_token=True)
            reconstruct_loss = reconstruct_loss.flatten()
            flattened_tokens = tokens[:, 1:].flatten()
            non_eos_mask = flattened_tokens != eos_token
            losses.append(reconstruct_loss[non_eos_mask].mean().item())
    return np.mean(losses)

def custom_reconstruct(
    enc: AutoEncoder, x: Float[Tensor, "batch d_in"], neuron: int | Tensor, activation: float | Tensor
):
    x_cent = x - enc.b_dec
    acts = F.relu(x_cent @ enc.W_enc + enc.b_enc)
    acts[:, neuron] = activation
    x_reconstruct = acts @ enc.W_dec + enc.b_dec
    return x_reconstruct

def custom_forward(
    enc: AutoEncoder, x: Float[Tensor, "batch d_in"], neuron: int | Tensor, activation: float | Tensor
):
    x_cent = x - enc.b_dec
    acts = F.relu(x_cent @ enc.W_enc + enc.b_enc)
    acts[:, neuron] = activation
    x_reconstruct = acts @ enc.W_dec + enc.b_dec
    l2_loss = (x_reconstruct - x).pow(2).sum(-1).mean(0)
    l1_loss = enc.reg_coeff * (acts.abs().sum())
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


def get_encode_activations_hook(
    encoder: AutoEncoder, encoder_neuron, cfg: AutoEncoderConfig, pos=-1
):
    def encode_activations_hook(value, hook):
        _, x_reconstruct, acts, _, _ = encoder(value[:, pos])
        feature_activations = acts[:, encoder_neuron]
        value[:, pos] = x_reconstruct
        return value

    return [(cfg.encoder_hook_point, encode_activations_hook)]


# def get_direction_logit_and_logprob_boost(
#     trigram_tokens: Int[Tensor, "batch pos"],
#     encoder: AutoEncoder,
#     encoder_neuron,
#     model: HookedTransformer,
#     trigram: str,
#     common_tokens: Int[Tensor, "tokens"],
#     all_ignore: Int[Tensor, "tokens"],
#     cfg: AutoEncoderConfig,
# ):
#     last_trigram_token = model.to_str_tokens(model.to_tokens(trigram))[-1]

#     zero_direction_hook = get_zero_ablate_encoder_direction_hook(
#         encoder, encoder_neuron, cfg
#     )
#     encode_activations_hook = get_encode_activations_hook(encoder, encoder_neuron, cfg)

#     with model.hooks(encode_activations_hook):
#         logits_active = model(trigram_tokens[:, :-1], return_type="logits")[:, -1]

#     with model.hooks(zero_direction_hook):
#         logits_inactive = model(trigram_tokens[:, :-1], return_type="logits")[:, -1]

#     last_trigram_token_tokenized = model.to_single_token(last_trigram_token)
#     last_token_logit_encoded = (
#         logits_active[:, last_trigram_token_tokenized].mean(0).item()
#     )
#     last_token_logit_zeroed = (
#         logits_inactive[:, last_trigram_token_tokenized].mean(0).item()
#     )
#     logging.info(
#         f"Logit '{last_trigram_token}' when reconstructing activations with encoder: {last_token_logit_encoded:.2f}"
#     )
#     logging.info(
#         f"Logit '{last_trigram_token}' when reconstructing activations with encoder zeroing out the direction: {last_token_logit_zeroed:.2f}"
#     )

#     logprobs_active = logits_active.log_softmax(dim=-1)
#     logprobs_inactive = logits_inactive.log_softmax(dim=-1)
#     last_token_logprob_encoded = (
#         logprobs_active[:, last_trigram_token_tokenized].mean(0).item()
#     )
#     last_token_logprob_zeroed = (
#         logprobs_inactive[:, last_trigram_token_tokenized].mean(0).item()
#     )
#     logging.info(
#         f"Logprob '{last_trigram_token}' when reconstructing activations with encoder: {last_token_logprob_encoded:.2f}"
#     )
#     logging.info(
#         f"Logprob '{last_trigram_token}' when reconstructing activations with encoder zeroing out the direction: {last_token_logprob_zeroed:.2f}"
#     )

#     boosts = (logprobs_active - logprobs_inactive).mean(0)
#     boosts[logprobs_active.mean(0) < -7] = 0
#     boosts[all_ignore] = 0
#     top_boosts, top_tokens = torch.topk(boosts, 15)
#     non_zero_boosts = top_boosts != 0
#     top_deboosts, top_deboosted_tokens = torch.topk(boosts, 15, largest=False)
#     non_zero_deboosts = top_deboosts != 0
#     boosted_tokens = (
#         model.to_str_tokens(top_tokens[non_zero_boosts]),
#         top_boosts[non_zero_boosts].tolist(),
#     )
#     deboosted_tokens = (
#         model.to_str_tokens(top_deboosted_tokens[non_zero_deboosts]),
#         top_deboosts[non_zero_deboosts].tolist(),
#     )
#     logging.info(f"Top boosted: {boosted_tokens}")
#     logging.info(f"Top deboosted: {deboosted_tokens}")

#     return (
#         last_token_logit_encoded,
#         last_token_logit_zeroed,
#         last_token_logprob_encoded,
#         last_token_logprob_zeroed,
#         boosted_tokens,
#         deboosted_tokens,
#     )


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

def get_custom_forward_hook(encoder, directions: int | Tensor, activation: float | Tensor, cfg: AutoEncoderConfig, pos=-2):
    def feature_hook(value, hook):
        x_reconstruct = custom_reconstruct(
            encoder, value[:, pos], directions, activation
        )
        value[:, pos] = x_reconstruct
        return value
    return [(cfg.encoder_hook_point, feature_hook)]

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


def generate_with_encoder(model: HookedTransformer, autoencoder: AutoEncoder, cfg: AutoEncoderConfig, input: str, k=20):
    def encode_activations_hook(value, hook):
        value = value.squeeze(0)
        _, x_reconstruct, _, _, _ = autoencoder(value)
        return x_reconstruct.unsqueeze(0)
    reconstruct_hooks = [(f'blocks.{cfg.layer}.{cfg.act_name}', encode_activations_hook)]

    with model.hooks(reconstruct_hooks):
        return model.generate(input, k, verbose=False, temperature=0, use_past_kv_cache=False)


def get_all_activating_test_prompts(prompts: list[str], encoder: AutoEncoder, model: HookedTransformer, cfg: AutoEncoderConfig, active_threshold=0.1, negative_pos_offset=-2):
    activating_test_prompts_all_dir = torch.zeros((len(prompts), encoder.d_hidden), dtype=torch.bool)
    for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        tokens = model.to_tokens(prompt)
        act_token_index = tokens.shape[1] + negative_pos_offset
        act = get_acts(prompt, model, encoder, cfg)[act_token_index]
        act_active = act > active_threshold
        activating_test_prompts_all_dir[i] = act_active
    return activating_test_prompts_all_dir

def eval_encoder_reconstruction_single_position(prompts, encoder: AutoEncoder, model: HookedTransformer, cfg: AutoEncoderConfig):
    """ Assumes prompts to end in answer tokens, evaluates reconstruction loss on the second to last token """
    # Original and zero ablation loss on test prompts
    original_losses = []
    ablated_losses = []
    encoded_losses = []

    def zero_ablate_mlp(value, hook):
        value[:, -2] = 0
        return value

    def encode_mlp(value, hook):
        recons = encoder(value[0, -2])[1]
        value[0, -2] = recons
        return value

    zero_ablate_mlp_hook = [(cfg.encoder_hook_point, zero_ablate_mlp)]
    encode_mlp_hook = [(cfg.encoder_hook_point, encode_mlp)]

    for prompt in tqdm(prompts):
        original_loss = model(prompt, return_type="loss", loss_per_token=True)[0, -1].item()
        with model.hooks(zero_ablate_mlp_hook):
            ablated_loss = model(prompt, return_type="loss", loss_per_token=True)[0, -1].item()
        with model.hooks(encode_mlp_hook):
            encoded_loss = model(prompt, return_type="loss", loss_per_token=True)[0, -1].item()
        original_losses.append(original_loss)
        ablated_losses.append(ablated_loss)
        encoded_losses.append(encoded_loss)

    original_loss = np.mean(original_losses)
    ablated_loss = np.mean(ablated_losses)
    encoded_loss = np.mean(encoded_losses)
    return original_loss, ablated_loss, encoded_loss

def get_top_direction_ablation_df(activating_test_prompts: Bool[Tensor, "n_test_prompts d_enc"], prompts: list[str], model, encoder: AutoEncoder, cfg: AutoEncoderConfig, max_activations: Float[Tensor, "n_data d_enc"]):
    """"""
    assert activating_test_prompts.shape[0] == len(prompts)

    all_acts = []
    for i in range(0, 1000, 32):
    # for prompt in prompts[:1000]:
        tokens = model.to_tokens(prompts[i:i + 32], padding_side='left')
        acts = get_acts(tokens, model, encoder, cfg)[: -2].tolist()
        all_acts.extend(acts)

    all_acts = torch.stack(all_acts)

    # Max per direction
    max_val, _ = max_activations.max(0)
    threshold_per_direction = (max_val * 0.17).cuda()

    # Mean activation on all prompts is misleading, prompts could be important on subset of test prompts
    num_active_acts = (all_acts > threshold_per_direction).sum(0) + 1e-9
    all_acts_tmp = all_acts.clone()
    all_acts_tmp[all_acts_tmp <= threshold_per_direction] = 0
    # Direction wise mean activation on active prompts
    mean_active_acts = all_acts_tmp.sum(0) / num_active_acts
    # Filter directions that are active on less than x% of quotation prompts
    mean_active_acts[num_active_acts < 0.02*all_acts.shape[0]] = 0
    n_non_zero_directions = (mean_active_acts > 0).sum().item()
    top_acts, top_dirs = torch.topk(mean_active_acts, min(100, n_non_zero_directions))
    
    print(f"Number of directions with mean activation > 0: {n_non_zero_directions}")

    # Run ablation for activating directions
    data = []
    global_loss_increases = []
    for direction in tqdm(top_dirs):
        hook = get_custom_forward_hook(encoder, direction, 0, cfg, pos=-2)
        loss_increases = []
        loss_increases_encoded_ablation = []
        active_test_prompt_indices = torch.argwhere(activating_test_prompts[:, direction]).flatten().tolist()
        active_test_prompts = [prompts[i] for i in active_test_prompt_indices]
        num_prompts = min(len(active_test_prompts), 200)
        for prompt in active_test_prompts[:num_prompts]:
            tokens = model.to_tokens(prompt, prepend_bos=False)
            pos = tokens.shape[1]-2
            original_loss, ablated_loss = evaluate_direction_ablation_single_prompt(tokens, encoder, model, direction.item(), cfg, pos=pos)
            with model.hooks(hook):
                encoded_ablated_loss = model(tokens, return_type="loss", loss_per_token=True)[0, -1].item()
            loss_increase = ablated_loss - original_loss
            loss_increases.append(loss_increase)
            loss_increases_encoded_ablation.append(encoded_ablated_loss - original_loss)
        loss_increase = np.mean(loss_increases)
        global_loss_increases.append(loss_increases)
        loss_increase_encoded_ablation = np.mean(loss_increases_encoded_ablation)
        # Summary statistic on subset of prompts
        mean_activation =  mean_active_acts[direction].item()
        percentage_activation = num_active_acts[direction].item() / all_acts.shape[0]
        data.append([direction.item(), loss_increase, loss_increase_encoded_ablation, mean_activation, percentage_activation])
    df = pd.DataFrame(data, columns=["Direction", "Loss increase", "Loss increase (encoded)", "Mean activation", "Percentage activation"])
    return df, global_loss_increases

def get_mean_component_wise_mlp(prompts, model, encoder_cfg):
    mlp_wise_decompositions = []
    for prompt in prompts:
        _, cache = model.run_with_cache(prompt)

        decomposition = cache.get_full_resid_decomposition(encoder_cfg.layer, mlp_input=True, apply_ln=True, return_labels=False, expand_neurons=False, pos_slice=None)
        decomposition = decomposition.squeeze(1) # Batch
        # Account for GELU in DLA by setting neuron contributions to 0 if they are not activated
        mlp_wise_decomposition = einops.einsum(decomposition, model.W_in[encoder_cfg.layer], "component pos d_res, d_res d_mlp -> component pos d_mlp")
        mlp_wise_decomposition = mlp_wise_decomposition.mean(1)
        mlp_wise_decompositions.append(mlp_wise_decomposition)
    mlp_wise_decompositions = torch.stack(mlp_wise_decompositions).mean(0)
    return mlp_wise_decompositions