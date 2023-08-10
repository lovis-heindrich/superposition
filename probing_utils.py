# All new word probe utils - not generalised

import torch
from transformer_lens import HookedTransformer
import plotly.io as pio
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef

from hook_utils import save_activation

pio.renderers.default = "notebook_connected+notebook"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)
torch.set_grad_enabled(False)

def get_new_word_labels(model: HookedTransformer, tokens: torch.Tensor) -> list[bool]:
    prompt_labels = []
    for i in range(tokens.shape[0] - 1):
        next_token_str = model.to_single_str_token(tokens[i + 1].item())
        next_is_space = next_token_str[0] in [" ", ",", ".", ":", ";", "!", "?"] # [" "] # 
        prompt_labels.append(next_is_space)
    return prompt_labels

def get_new_word_labels_and_activations(
    model: HookedTransformer, 
    german_data: list[str], 
    activation_hook_name: str,
    activation_slice=np.s_[0, :-1, 2994:2995]
) -> tuple[np.ndarray, np.ndarray]:
    '''Get activations and labels for predicting word end from activation'''
    activations = []
    labels = []
    # positive_activations = []
    # positive_labels = []
    # negative_activations = []
    # negative_labels = []
    for prompt in german_data:
        tokens = model.to_tokens(prompt)[0]

        with model.hooks([(activation_hook_name, save_activation)]):
            model(tokens)
        prompt_activations = model.hook_dict[activation_hook_name].ctx['activation']
        prompt_activations = prompt_activations[activation_slice].cpu().numpy()
        activations.append(prompt_activations)

        prompt_labels = get_new_word_labels(model, tokens)
        labels.extend(prompt_labels)

    
    activations = np.concatenate(activations, axis=0)
    labels = np.array(labels) # , axis=0

    assert activations.shape[0] == labels.shape[0]
    return activations, labels

def get_probe(x: np.ndarray, y: np.ndarray) -> float:
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)
    # np.unique on y 
    lr_model = LogisticRegression(max_iter=2000)
    lr_model.fit(x, y)
    return lr_model

def get_probe_score(lr_model: LogisticRegression, x, y) -> float:
    # z-scoring can help with convergence
    preds = lr_model.predict(x)
    f1 = f1_score(y, preds)
    mcc = matthews_corrcoef(y, preds)
    return f1, mcc

def get_and_score_new_word_probe(
    model: HookedTransformer, 
    german_data: list[str], 
    activation_hook_name: str,
    activation_slice=np.s_[0, :-1, 2994:2995]
):
    x, y = get_new_word_labels_and_activations(model, german_data, activation_hook_name, activation_slice)
    print(y.shape, "items")
    probe = get_probe(x[:20_000], y[:20_000])
    f1, mcc = get_probe_score(probe, x[20_000:], y[20_000:])
    return f1, mcc