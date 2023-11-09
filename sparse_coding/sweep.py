# Sweep over autoencoder configurations. To change more than one setting per run, define a configuration 
# file for each model in the sweep then set the param to "cfg_file" and values to a list of file paths

import argparse
import json
from transformer_lens import HookedTransformer
import torch
import numpy as np
from random import random

from process_tiny_stories_data import (
    load_tinystories_validation_prompts,
    load_tinystories_tokens,
)
from sparse_coding.train_autoencoder import get_autoencoder, main, DEFAULT_CONFIG
from utils.autoencoder_utils import AutoEncoderConfig
from utils.haystack_utils import get_device


def set_seed(cfg: AutoEncoderConfig) -> int:
    seed = cfg["seed"]
    np.random.seed(seed)
    random.seed(seed)
    return seed


def get_config(param: str, value: bool | float | int | str):
    cfg = DEFAULT_CONFIG.copy()
    cfg[param] = value

    # Accept alternative config values from file specified in command line
    if cfg["cfg_file"] is not None:
        with open(cfg["cfg_file"], "r") as f:
            cfg.update(json.load(f))

    # Derive config values
    cfg["model_batch_size"] = cfg["batch_size"] // cfg["seq_len"] * 16
    cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
    cfg["buffer_batches"] = (
        cfg["buffer_size"] // cfg["seq_len"]
    )  # (cfg["model_batch_size"] * cfg["seq_len"])
    cfg["num_eval_batches"] = cfg["num_eval_tokens"] // cfg["batch_size"]
    assert (
        cfg["buffer_batches"] % cfg["model_batch_size"] == 0
    ), "Buffer batches must be multiple of model batch size"

    return cfg

if __name__ == "__main__":
    torch.set_grad_enabled(True)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(f"--param", type=str, default=None)
    parser.add_argument(f"--values", type=list, default=None)
    args = parser.parse_args()
    assert (
        args.param is not None and args.values is not None
    ), "Must define sweep's varying parameter and values, e.g. --param seed --values ['3', '4']"

    device = get_device()
    num_devices = torch.cuda.device_count()

    for value in args.values:
        torch.cuda.empty_cache()

        cfg = get_config(args.param, args.value)
        seed = set_seed(cfg)
        
        prompt_data = load_tinystories_tokens(cfg["data_path"])
        eval_prompts = load_tinystories_validation_prompts(cfg["data_path"])[
            : cfg["num_eval_prompts"]
        ]
        
        model = HookedTransformer.from_pretrained(
            cfg["model"],
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            device=device,
        )
        cfg["d_mlp"] = model.cfg.d_mlp

        encoder = get_autoencoder(cfg, device, seed)
        
        main(encoder, model, cfg, prompt_data, eval_prompts)
