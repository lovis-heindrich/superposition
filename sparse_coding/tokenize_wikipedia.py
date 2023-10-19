import tarfile
import re
import json
from tqdm import tqdm
import os
from datasets import load_dataset

# languages = ["bg", "cs", "da", "de", "el", "es", "en", "et", "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]
wikipedia_languages = ["en", "de"]
datasets = {lang: [] for lang in wikipedia_languages}


def add_to_dataset(text: str, language: str, min_chars=500):
    '''Get lines of sufficient length and add to the global datasets dictionary'''
    # data = preprocess_data(text)
    for line in tqdm(data):
        if len(line['text']) > min_chars:
            datasets[language].append(line['text'])

    with open(f'data/wikipedia/{language}_samples.json', 'w', encoding='utf-8') as f:
        json.dump(datasets[language], f)
    print(f'{language}: {len(datasets[language])}')
    
if __name__ == "__main__":
    os.makedirs('data/wikipedia', exist_ok=True)

    for language in wikipedia_languages:
        data = load_dataset("wikipedia", f"20220301.{language}", split="train")
        add_to_dataset(data, language)