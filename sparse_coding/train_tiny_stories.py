import setup

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AdamW, get_scheduler, Trainer, TrainingArguments, GPTNeoForCausalLM, GPT2Tokenizer, GPTNeoConfig, GPTNeoModel, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset

from utils.haystack_utils import load_json_data


dataset = load_dataset("roneneldan/TinyStories")
config = load_json_data('sparse_coding/config/tiny-stories-2L-33M_model.json')

def tokenize_function(examples):
    examples = tokenizer(examples["text"], padding="max_length", truncation=True)
    examples["input_ids"] = examples.data['input_ids']
    examples["labels"] = examples["input_ids"].copy()
    return examples

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.bos_token

# needs cols required in model forward
tokenized_datasets = dataset.map(tokenize_function, batched=True)

config = GPTNeoConfig.from_json_file('sparse_coding/config/tiny-stories-2L-33M_model.json')
model = GPTNeoForCausalLM(config)

training_args = TrainingArguments(
    output_dir="./gpt-neo-tinystories",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()

model.save_pretrained("./gpt-neo-tinystories")