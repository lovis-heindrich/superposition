# Sweep over autoencoder configurations. To change more than one setting per run, define a configuration 
# file for each model in the sweep then set the param to "cfg_file" and values to a list of file paths
import sys
import os
sys.path.append('../')
import argparse
import subprocess
import torch
import wandb

from sparse_coding.train_autoencoder import DEFAULT_CONFIG


def define_sweep(param: str, values: list[float | int | bool | str]):
    cfg = DEFAULT_CONFIG.copy()
    sweep_config = {
        'method': 'grid',
        'parameters': {
            key: {'values': [value]} for key, value in cfg.items()
        }
    }
    sweep_config['parameters'][param] = {'values': values}
    sweep_id = wandb.sweep(sweep_config, project=f'{cfg["model"]}-{cfg["layer"]}-autoencoder')
    return sweep_id

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-p', "--param", type=str, required=True)
    parser.add_argument('-v','--values', nargs='+', type=int, required=True)
    args = parser.parse_args()
    assert (
        args.param is not None and args.values is not None
    ), "Must define sweep's varying parameter and values, e.g. --param seed --values ['3', '4']"

    sweep_id = define_sweep(args.param, args.values)
    for device_index in range(torch.cuda.device_count()):
        cmd = f"CUDA_VISIBLE_DEVICES={device_index} wandb agent {sweep_id} --count 5 --entity wandb_sweep_train.py"
        subprocess.run(cmd, shell=True)