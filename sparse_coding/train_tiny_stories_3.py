os.environ['TRANSFORMERS_CACHE'] = "/workspace"

import os
import pickle
import torch
import json
from tqdm.auto import tqdm
import wandb
from torch.utils.data import DataLoader
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GPTNeoConfig
from datasets import load_dataset
from utils.haystack_utils import load_json_data, clean_cache

clean_cache()
model_name = 'tiny-stories-33M'
device = "cuda"

dataset = load_dataset("roneneldan/TinyStories")
cfg = load_json_data(f'/workspace/config/{model_name}_model.json')
cfg["model"] = model_name

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M", cache_dir="/workspace")
print("Loaded tokenizer")
tokenizer.pad_token = tokenizer.bos_token


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", max_length=cfg["window_size"], truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    labels = input_ids.clone()
    return input_ids, labels

train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=cfg["batch_size"], collate_fn=collate_fn)

config = GPTNeoConfig.from_json_file(f"/workspace/config/{cfg['model']}_model.json")
model = GPTNeoForCausalLM(config).to(device)
print("Loaded model")
optim = torch.optim.AdamW(
    model.parameters(),
    lr=cfg["lr"],
    betas=(cfg["beta1"], cfg["beta2"]),
    weight_decay=cfg["wd"]
)

if cfg["use_wandb"]:
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

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
model.train()
print("Training...")
for epoch in tqdm(range(cfg["epochs"])):
    for i, (input_ids, labels) in tqdm(enumerate(train_dataloader)):
        input_ids = input_ids.to(device)
        optim.zero_grad()
        outputs = model(input_ids[:, :-1], labels=input_ids[:, 1:])
        loss = outputs.loss
        loss.backward()
        optim.step()

        if cfg["use_wandb"]:
            log_dict = {
                "batch": i,
                "loss": loss.item(),
                "epoch": epoch,
            }
            wandb.log(log_dict)

        if i % 10_000 == 0:
            torch.save(model.state_dict(), f"{cfg['save_path']}/{cfg['model']}/{save_name}_{i}.pt")
