import tarfile
import re
import json

# Assumes you have downloaded your own copy of europarl.tgz from https://www.statmt.org/europarl/

europarl_languages = ["bg", "cs", "da", "de", "el", "es", "en", "et", "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]
datasets = {lang: [] for lang in europarl_languages}


def preprocess_data(text: str):
    '''Roughly remove the metadata lines, e.g. <Chapter 1> and <Speaker 1>'''
    return re.sub(r'<.*>', '', text)


def add_to_dataset(data: str, language: str, num_samples=200, min_chars=500):
    '''Get the first 200 lines of sufficient length and save to a dataset file'''
    for line in data.split('\n'):
        if len(line) > min_chars:
            datasets[language].append(line)
            if len(datasets[language]) >= num_samples:
                with open(f'{language}_{num_samples}_samples.json', 'w', encoding='utf-8') as f:
                        json.dump(datasets[language], f)
                europarl_languages.remove(language)
                break


with tarfile.open('europarl.tgz', 'r:gz') as tar:
    for i, member in enumerate(tar.getmembers()):
        if not member.isfile():
            continue
        lang = member.name.split('/')[1]
        if lang in europarl_languages:
            file = tar.extractfile(member)
            text = file.read().decode('utf-8')
            data = preprocess_data(text)
            add_to_dataset(data, lang)

    print("Languages with insufficient data:", europarl_languages)
            