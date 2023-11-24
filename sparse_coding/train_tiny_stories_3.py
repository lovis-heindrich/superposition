import os
import torch
import json
import argparse
import numpy as np
import sys
sys.path.append('./')
sys.path.append('../')
# import setup
from tqdm.auto import tqdm
import wandb
from torch.utils.data import DataLoader
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GPTNeoConfig
from datasets import load_dataset
from utils.haystack_utils import load_json_data, clean_cache
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from sparse_coding.process_tiny_stories_data import load_tinystories_validation_prompts


cache_dir = '/workspace/cache'
os.makedirs(cache_dir, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["WANDB_SILENT"] = "true"


def train():
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    n = torch.cuda.device_count() // local_world_size
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
    )

    model_name = 'tiny-stories-33M'
    cfg = load_json_data(f'/workspace/config/{model_name}_model.json')
    cfg["model"] = model_name

    dataset = load_dataset("roneneldan/TinyStories")

    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.bos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=cfg["window_size"], truncation=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    def collate_fn(batch):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        labels = input_ids.clone()[:, 1:]
        return input_ids[:, :-1], labels

    train_sampler = DistributedSampler(tokenized_datasets["train"])
    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=cfg["batch_size"], collate_fn=collate_fn, sampler=train_sampler)

    config = GPTNeoConfig.from_json_file(f"/workspace/config/{cfg['model']}_model.json")
    model = GPTNeoForCausalLM(config).cuda(device_ids[0])
    model = DDP(model, device_ids)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=cfg["wd"]
    )
    validation_prompts = load_tinystories_validation_prompts()[:100]
    tokenized_validation = [tokenizer(prompt, padding="max_length", max_length=cfg["window_size"], truncation=True) for prompt in validation_prompts]

    if cfg["use_wandb"] and local_rank == 0:
        wandb.init(project=f'{cfg["model"]}', config=cfg)
        wandb_name = wandb.run.name
        save_name = f"{wandb_name.split('-')[-1]}_" + "_".join(
            wandb_name.split("-")[:-1]
        )
        cfg["wandb_name"] = wandb_name
    else:
        save_name = "local"
    cfg["save_name"] = save_name
    os.makedirs(f"{cfg['save_path']}/{cfg['model']}", exist_ok=True)
    with open(f"{cfg['save_path']}/{cfg['model']}/{save_name}.json", "w") as f:
        json.dump(cfg, f, indent=4)

    model.train()
    for epoch in range(cfg["epochs"]):
        train_sampler.set_epoch(epoch)
        for i, (input_ids, labels) in tqdm(enumerate(train_dataloader)):
            input_ids = input_ids.to(f"cuda:{n}")
            optim.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()

            with torch.no_grad():
                log_dict = {
                    "batch": i,
                    "loss": loss.item(),
                    "epoch": epoch,
                }

                if i % (10_000 // cfg['batch_size']) == 0 and local_rank == 0:
                    torch.save(model.state_dict(), f"{cfg['save_path']}/{cfg['model']}/{save_name}_{i}.pt")

                    reconstruction_losses = []
                    for tokenized_prompt in tokenized_validation:
                        tokens = torch.tensor(tokenized_prompt['input_ids'])
                        reconstruction_losses.append(model(tokens[:-1], labels=tokens[1:]).loss.item())
                        log_dict['reconstruction_loss'] = np.mean(reconstruction_losses)

                if cfg["use_wandb"] and local_rank == 0:  
                    wandb.log(log_dict)
                    

    torch.save(model.state_dict(), f"{cfg['save_path']}/{cfg['model']}/{save_name}.pt")


@record
def main():
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    
    dist.init_process_group(backend="nccl")
    train()
    dist.destroy_process_group()

    
if __name__ == "__main__":
    main()