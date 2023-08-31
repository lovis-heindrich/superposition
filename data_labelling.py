import csv
import os
import pickle
from datasets import load_dataset
from transformer_lens import HookedTransformer


DATA_PATH = 'data/gpt2_large_spectrum.pkl'
LABEL_PATH = 'data/gp2_large_spectrum_labels.csv'

def load_text():
    text = load_dataset("stas/openwebtext-10k", split="train")
    return [item for i, item in enumerate(text["text"]) if len(item) > 2000 and i < 5000]

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
    text = load_text()
    df = read_data(DATA_PATH)
    labels = read_labels(LABEL_PATH)
    print(len(labels.keys()), 'labels')
    for i in range(len(df)):
        if str(i) in labels:
            continue

        prompt_index = df.loc[i, 'prompt_index']
        tokens = model.to_tokens(text[prompt_index])[0]

        token_index = df.loc[i, 'token_index']
        token = tokens[token_index].item()

        if tokens[token_index + 1].item() in an_tokens:
            print('Skipping \' an\'')
            write_label(LABEL_PATH, i, '3')
            continue

        starting_index = max(0, token_index - 50)
        str_tokens = model.to_str_tokens(tokens[starting_index:token_index + 2])
        str_tokens[-1] = CYAN_COL + str_tokens[-1] + END_COL
        print(''.join(str_tokens))
        label = input(f'Label for token after {model.to_single_str_token(token)}, enter 1 for implausible \' an\', or 2 for plausible \' an\':')
        while label not in ['1', '2']:
            label = input('Invalid input. Enter 1 for implausible \' an\', or 2 for plausible \' an\':')
        write_label(LABEL_PATH, i, label)

if __name__ == '__main__':
    main()
