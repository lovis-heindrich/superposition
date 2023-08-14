# All new word probe utils - not generalised

from collections import defaultdict
from functools import partial

import torch
from transformer_lens import HookedTransformer
import plotly.io as pio
import pandas as pd
import numpy as np
from concept_erasure import LeaceEraser

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.utils import shuffle

from hook_utils import save_activation

pio.renderers.default = "notebook_connected+notebook"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)

def get_act_dimension(model, prompt, activation_hook_name, activation_slice):
    tokens = model.to_tokens(prompt)[0]
    with model.hooks([(activation_hook_name, save_activation)]):
        model(tokens)
    prompt_activations = model.hook_dict[activation_hook_name].ctx['activation'][activation_slice]
    return prompt_activations.shape[-1]

def get_new_word_labels(model: HookedTransformer, tokens: torch.Tensor) -> np.ndarray[bool]:
    prompt_labels = []
    for i in range(tokens.shape[0] - 1):
        next_token_str = model.to_single_str_token(tokens[i + 1].item())
        next_is_space = next_token_str[0] in [" ", ",", ".", ":", ";", "!", "?"] # [" "] # 
        prompt_labels.append(next_is_space)
    return np.array(prompt_labels)

def get_new_word_labels_and_activations(
    model: HookedTransformer, 
    german_data: list[str], 
    activation_hook_name: str,
    activation_slice=np.s_[0, :-1, 2994:2995],
    num_class_examples=20_000, scale_x=True
) -> tuple[np.ndarray, np.ndarray]:
    '''Get activations and labels for predicting word end from activation'''
    x_dimension = get_act_dimension(model, german_data[0], activation_hook_name, activation_slice)
    positive_activations = np.empty((num_class_examples, x_dimension))
    negative_activations = np.empty((num_class_examples, x_dimension))
    positive_start = 0
    negative_start = 0
    for prompt in german_data:
        tokens = model.to_tokens(prompt)[0]
        labels = get_new_word_labels(model, tokens)

        with model.hooks([(activation_hook_name, save_activation)]):
            model(tokens)
        prompt_activations = model.hook_dict[activation_hook_name].ctx['activation']
        prompt_activations = prompt_activations[activation_slice].cpu().numpy()
        
        if positive_start < num_class_examples:
            positive_end = min(positive_start + prompt_activations[labels].shape[0], num_class_examples)
            positive_activations[positive_start:positive_end] = prompt_activations[labels][:positive_end - positive_start]
            positive_start = positive_end
        if negative_start < num_class_examples:
            negative_end = min(negative_start + prompt_activations[~labels].shape[0], num_class_examples)
            negative_activations[negative_start:negative_end] = prompt_activations[~labels][:negative_end - negative_start]
            negative_start = negative_end
        if positive_start >= num_class_examples and negative_start >= num_class_examples:
            break

    x = np.concatenate((positive_activations, negative_activations))
    y = np.concatenate((np.full(positive_activations.shape[0], True), np.full(negative_activations.shape[0], False)))
    if scale_x:
        scaler = preprocessing.StandardScaler().fit(x)
        x = scaler.transform(x)
    return shuffle(x, y)

def get_new_word_labels_and_resid_activations(
    model: HookedTransformer, 
    german_data: list[str], 
    num_class_examples=20_000, scale_x=True
) -> tuple[defaultdict, np.ndarray]:
    '''Get activations and labels for predicting word end from activation'''
    _, cache = model.run_with_cache(model.to_tokens(german_data[0])[0])
    resid = cache.decompose_resid(apply_ln=False)[:, 0, :-1, :]
    
    positive_activations = {i: np.empty((num_class_examples, model.cfg.d_model)) for i in range(resid.shape[0])}
    negative_activations = {i: np.empty((num_class_examples, model.cfg.d_model)) for i in range(resid.shape[0])}
    positive_start = 0
    negative_start = 0
    for prompt in german_data:
        tokens = model.to_tokens(prompt)[0]
        labels = get_new_word_labels(model, tokens)
        _, cache = model.run_with_cache(tokens)
        resid = cache.decompose_resid(apply_ln=False)[:, 0, :-1, :]

        for i in range(resid.shape[0]):
            acts = resid[i].cpu().numpy()
            if positive_start < num_class_examples:
                positive_end = min(positive_start + acts[labels].shape[0], num_class_examples)
                positive_activations[i][positive_start:positive_end] = acts[labels][:positive_end - positive_start]
                positive_start = positive_end
            if negative_start < num_class_examples:
                negative_end = min(negative_start + acts[~labels].shape[0], num_class_examples)
                negative_activations[i][negative_start:negative_end] = acts[~labels][:negative_end - negative_start]
                negative_start = negative_end
        if positive_start >= num_class_examples and negative_start >= num_class_examples:
            break

    x = {}
    y = {}
    for i in range(resid.shape[0]):
        activations = np.concatenate((positive_activations[i], negative_activations[i]))
        labels = np.concatenate((np.full(positive_activations[i].shape[0], True), np.full(negative_activations[i].shape[0], False)))
        if scale_x:
            scaler = preprocessing.StandardScaler().fit(activations)
            activations = scaler.transform(activations)
        x[i], y[i] = shuffle(activations, labels)
    return x, y

def get_is_german_labels_and_resid_activations(
    model: HookedTransformer, 
    german_data: list[str], 
    english_data: list[str],
    layer=8,
    num_class_examples=20_000, scale_x=True
) -> tuple[defaultdict, np.ndarray]:
    '''Get activations and labels for predicting word end from activation'''
    positive_activations = np.empty((num_class_examples, model.cfg.d_model))
    start = 0
    for prompt in german_data:
        tokens = model.to_tokens(prompt)[0]
        _, cache = model.run_with_cache(tokens)
        acts = cache[f'blocks.{layer}.hook_resid_pre'][0].cpu().numpy()


        if start < num_class_examples:
            end = min(start + acts.shape[0], num_class_examples)
            positive_activations[start:end] = acts[:end - start]
            start = end
        if start >= num_class_examples:
            break

    negative_activations = np.empty((num_class_examples, model.cfg.d_model))
    start = 0
    for prompt in english_data:
        tokens = model.to_tokens(prompt)[0]
        _, cache = model.run_with_cache(tokens)
        acts = cache[f'blocks.{layer}.hook_resid_pre'][0].cpu().numpy()

        if start < num_class_examples:
            end = min(start + acts.shape[0], num_class_examples)
            negative_activations[start:end] = acts[:end - start]
            start = end
        if start >= num_class_examples:
            break
        
    activations = np.concatenate((positive_activations, negative_activations))
    labels = np.concatenate((np.full(positive_activations.shape[0], True), np.full(negative_activations.shape[0], False)))
    if scale_x:
        scaler = preprocessing.StandardScaler().fit(activations)
        activations = scaler.transform(activations)
    x, y = shuffle(activations, labels)
    return x, y

def get_probe(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    # z-scoring can help with convergence
    #scaler = preprocessing.StandardScaler().fit(x)
    #x = scaler.transform(x)
    # np.unique on y 
    lr_model = LogisticRegression(max_iter=2000)
    lr_model.fit(x, y)
    return lr_model

def get_probe_score(lr_model: LogisticRegression, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    preds = lr_model.predict(x)
    f1 = f1_score(y, preds)
    mcc = matthews_corrcoef(y, preds)
    return f1, mcc

def get_and_score_new_word_probe(
    model: HookedTransformer, 
    german_data: list[str], 
    activation_hook_name: str,
    # The final position's activation has no label and is excluded
    activation_slice=np.s_[0, :-1, 2994:2995]
):
    x, y = get_new_word_labels_and_activations(model, german_data, activation_hook_name, activation_slice)
    probe = get_probe(x[:20_000], y[:20_000])
    f1, mcc = get_probe_score(probe, x[20_000:], y[20_000:])
    return f1, mcc


def get_leace_eraser(x: np.ndarray, y: np.ndarray):
    X_t = torch.from_numpy(x)
    Y_t = torch.from_numpy(~y)
    eraser = LeaceEraser.fit(X_t, Y_t)
    return eraser