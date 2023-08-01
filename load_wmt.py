# %%
from collections import Counter
import json
import requests
from tqdm import tqdm
import os

# %%


def download_dataset(url, file_path):
    # Create necessary directories if not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Send a HTTP request to the URL
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    # Initialize a progress bar
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    # Save the content in a .zst file
    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            # Update the progress bar
            progress_bar.update(len(chunk))
            file.write(chunk)

    progress_bar.close()


# Use the function
url = "https://the-eye.eu/public/AI/pile_v2/data/EuroParliamentProceedings_1996_2011.jsonl.zst"
file_name = "./data/pile/EuroParliamentProceedings_1996_2011.jsonl.zst"
download_dataset(url, file_name)

# %%

import zstandard as zstd
from pathlib import Path
import json
import io

DCTX = zstd.ZstdDecompressor(max_window_size=2**31)

def read_lines_from_zst_file(zstd_file_path:Path):
    with (
        zstd.open(zstd_file_path, mode='rb', dctx=DCTX) as zfh,
        io.TextIOWrapper(zfh) as iofh
    ):
        for line in iofh:
            yield line

file = Path(file_name)
records = map(json.loads, read_lines_from_zst_file(file))
# %%

language_counter = Counter()
for record in tqdm(records):
    language_counter[record["meta"]["language"]] += 1

print(language_counter)
# %%

languages = ["de", "nl", "it", "en", "es", "sv"]

language_data = {lang:[] for lang in languages}
for record in tqdm(records):
    language = record["meta"]["language"]
    if language in languages:
        language_data[language].append(record)
# %%
import pickle
with open("./data/wmt_europarl.pkl", "wb") as f:
    pickle.dump(language_data, f)
# %%
def print_file_size_in_mb(file_path):
    size_in_bytes = os.path.getsize(file_path)
    size_in_mb = size_in_bytes / (1024 * 1024)
    print(f"File size: {size_in_mb} MB")

# Use the function
file_path = "./data/wmt_europarl.pkl"
print_file_size_in_mb(file_path)
# %%
