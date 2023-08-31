import csv
import os
import pickle
import numpy as np
from datasets import load_dataset
from transformer_lens import HookedTransformer
import torch

DATA_PATH = 'data/gpt2_large_spectrum_v2.pkl'
LABEL_PATH = 'data/gp2_large_spectrum_labels_v2.csv'

def load_text():
    text = load_dataset("stas/openwebtext-10k", split="train")
    return text['text']

def read_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def read_labels(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return {row[0]: row[1] for row in csv.reader(f)}
    return {}

def write_label(file_path, text, label):
    with open(file_path, 'a') as f:
        csv.writer(f).writerow([text, label])

END_COL = '\033[0m'
CYAN_COL = '\033[96m'
def main():
    model = HookedTransformer.from_pretrained("gpt2-large")
    an_tokens = [
        model.to_single_token(' an'), 
        model.to_single_token(' An'), 
        model.to_single_token(' AN')
        ]
    plausible_pre_and_tokens = [
        model.to_single_token(' about'),
        model.to_single_token(' around'),
        model.to_single_token(' have'),
        model.to_single_token(' in'),
        model.to_single_token(' is'),
        model.to_single_token(' of'),
        model.to_single_token(' at'),
        model.to_single_token(' am'),
        model.to_single_token(' to'),
        model.to_single_token(' put'),
        model.to_single_token(' than'),
        model.to_single_token(' half'),
        model.to_single_token(' was'),
        model.to_single_token(' with'),
        model.to_single_token(' within'),
        model.to_single_token(' said'),
        model.to_single_token(' considering'),
        model.to_single_token(' using'),
        model.to_single_token(' keep'),
        model.to_single_token(' Keep'),
        model.to_single_token(' me'),
        model.to_single_token(' About'),
        model.to_single_token('roximately'),
    ]
    text = load_text()
    df = read_data(DATA_PATH)
    indices = np.arange(len(df))
    df['row_index'] = indices
    df = df.sample(frac=1).reset_index(drop=True)

    labels = read_labels(LABEL_PATH)
    print(len(labels.keys()), 'labels')
    for i in range(len(df)):
        index = df.loc[i, 'row_index']
        if str(index) in labels:
          continue

        prompt_index = df.loc[i, 'prompt_index']
        tokens = model.to_tokens(text[prompt_index])[0]

        token_index = df.loc[i, 'token_index']
        token = tokens[token_index].item()

        if token in plausible_pre_and_tokens:
            print('Skipping \' an\' after plausible preposition')
            write_label(LABEL_PATH, index, '2')
            continue

        if tokens[token_index + 1].item() in an_tokens:
            print('Skipping \' an\'')
            write_label(LABEL_PATH, index, '3')
            continue

        starting_index = max(0, token_index - 50)
        str_tokens = model.to_str_tokens(tokens[starting_index:token_index + 1])
        str_tokens[-1] = CYAN_COL + str_tokens[-1] + END_COL
        print(''.join(str_tokens))
        label = input(f'Label for token after {model.to_single_str_token(token)}, enter 1 for implausible \' an\', or 2 for plausible \' an\':')
        while label not in ['1', '2']:
            label = input('Invalid input. Enter 1 for implausible \' an\', or 2 for plausible \' an\':')
        write_label(LABEL_PATH, index, label)

if __name__ == '__main__':
    main()
