# Cancelled as Pile data isn't labelled with language (see note in paper)

import json
import pandas as pd
import zstandard
import os
import tqdm

file_path = 'process_data/00.jsonl.zst'

def read_zst_jsonl(file_path):
        with open(file_path, 'rb') as fh:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text = ""
                while True:
                    chunk = reader.read(65536)
                    if not chunk:
                        break
                    text += chunk.decode('utf-8')
                    
                    while "\n" in text:
                        line, text = text.split("\n", 1)
                        yield json.loads(line)

def main():
    europarl_languages = ["bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]
    datasets = {lang: [] for lang in europarl_languages}

    for item in tqdm.tqdm(read_zst_jsonl(file_path)):
        lang = item.get('meta', {}).get('language')
        text = item.get('text')

        if 'parl' in item.get('meta', {}):
            print(item.get('meta', {}))
        
        if lang in europarl_languages:
            if len(text) > 500:
                datasets[lang].append(text)
                
                if len(datasets[lang]) >= 200:
                    europarl_languages.remove(lang)
        
        if not europarl_languages:
            break

    print("Languages without a full dataset: ", europarl_languages)
    for lang, data in datasets.items():
        df = pd.DataFrame(data, columns=['text'])
        df.to_csv(f"data/languages/{lang}_dataset.csv", index=False)

if __name__ == "__main__":
    main()