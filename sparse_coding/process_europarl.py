import tarfile
import re
import json
from tqdm import tqdm

# Assumes you have downloaded your own copy of europarl.tgz from https://www.statmt.org/europarl/

# europarl_languages = ["bg", "cs", "da", "de", "el", "es", "en", "et", "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]
europarl_languages = ["en", "de"]
datasets = {lang: [] for lang in europarl_languages}


def preprocess_data(text: str):
    '''Roughly remove the metadata lines, e.g. <Chapter 1> and <Speaker 1>'''
    return re.sub(r'<.*>', '', text)

def add_to_dataset(text: str, language: str, min_chars=500):
    '''Get the first 200 lines of sufficient length and save to a dataset file'''
    data = preprocess_data(text)
    for line in data.split('\n'):
        if len(line) > min_chars:
            datasets[language].append(line)
    
if __name__ == "__main__":
    with tarfile.open('data/europarl.tgz', 'r:gz') as tar:
        for i, member in tqdm(enumerate(tar.getmembers())):
            if not member.isfile():
                continue
            lang = member.name.split('/')[1]
            if lang in europarl_languages:
                file = tar.extractfile(member)
                text = file.read().decode('utf-8')
                add_to_dataset(text, lang)
    for language in europarl_languages:
        print(f'{language}: {len(datasets[language])}')
        with open(f'data/europarl/{language}_samples.json', 'w', encoding='utf-8') as f:
            json.dump(datasets[language], f)
