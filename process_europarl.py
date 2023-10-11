import tarfile
import re
import json

# Assumes you have downloaded your own copy of europarl.tgz from https://www.statmt.org/europarl/

europarl_languages = ["bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]
datasets = {lang: [] for lang in europarl_languages}

def preprocess_data(text):
    '''Roughly remove the metadata lines, e.g. <Chapter 1> and <Speaker 1>'''
    result = []
    for line in text.split('\n'):
        line = re.sub(r'<.*>', '', line)
        if line:
            result.append(line)
    return '\n'.join(result)


def create_dataset(text, language, num_samples=200, min_chars=500):
    '''Get the first 200 lines of sufficient length and save to a dataset file'''
    content = preprocess_data(text.read().decode('utf-8'))
    for line in content.split('\n'):
        if len(line) > min_chars:
            datasets[language].append(line)
            if len(datasets[language]) >= num_samples:
                with open(f'{language}_200_samples.txt', 'w', encoding='utf-8') as f:
                        json.dump('\n'.join(datasets[language]), f)
                europarl_languages.remove(language)
                break
    

with tarfile.open('europarl.tgz', 'r:gz') as tar:
    for i, member in enumerate(tar.getmembers()):
        if not member.isfile():
            continue
        lang = member.name.split('/')[1]
        if lang in europarl_languages:
            file = tar.extractfile(member)
            create_dataset(file, lang)
            