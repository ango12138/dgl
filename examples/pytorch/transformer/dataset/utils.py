import numpy as np
import requests
import torch as th
import os
from tqdm import tqdm

def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    def process_response(r):
        chunk_size = 16 * 1024
        total_size = int(r.headers.get('Content-length', 0))
        with open(path, "wb") as file:
            with tqdm(total=total_size, unit='B',
                      unit_scale=1, desc=path.split('/')[-1]) as t:
                for chunk in r.iter_content(chunk_size):
                    if chunk:
                        file.write(chunk)
                        t.update(len(chunk))

    if 'drive.google.com' not in url:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
        process_response(response)
        return

    print('downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    process_response(response)

def prepare_dataset(dataset_name):
    directory = os.path.join('data', dataset_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        return
    if dataset_name == 'multi30k':
        os.system('bash scripts/prepare-multi30k.sh')
    elif dataset_name == 'wmt14':
        download_from_url('https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8', 'wmt16_en_de.tar.gz')
        os.system('bash scripts/prepare-wmt14.sh')
    elif dataset_name == 'copy':
        train_size = 9000
        valid_size = 1000
        test_size = 1000
        char_list = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        with open(os.path.join(directory, 'train.in'), 'w') as f_in,\
            open(os.path.join(directory, 'train.out'), 'w') as f_out:
            for i, l in zip(range(train_size), np.random.normal(15, 3, train_size).astype(int)):
                l = max(l, 1)
                line = ' '.join(np.random.choice(char_list, l)) + '\n'
                f_in.write(line)
                f_out.write(line)

        with open(os.path.join(directory, 'valid.in'), 'w') as f_in,\
            open(os.path.join(directory, 'valid.out'), 'w') as f_out:
            for i, l in zip(range(valid_size), np.random.normal(15, 3, valid_size).astype(int)):
                l = max(l, 1)
                line = ' '.join(np.random.choice(char_list, l)) + '\n'
                f_in.write(line)
                f_out.write(line)

        with open(os.path.join(directory, 'test.in'), 'w') as f_in,\
            open(os.path.join(directory, 'test.out'), 'w') as f_out:
            for i, l in zip(range(test_size), np.random.normal(15, 3, test_size).astype(int)):
                l = max(l, 1)
                line = ' '.join(np.random.choice(char_list, l)) + '\n'
                f_in.write(line)
                f_out.write(line)

        with open(os.path.join(directory, 'vocab.txt'), 'w') as f:
            for c in char_list:
                f.write(c + '\n')

    elif dataset_name == 'sort':
        train_size = 9000
        valid_size = 1000
        test_size = 1000
        char_list = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        with open(os.path.join(directory, 'train.in'), 'w') as f_in,\
            open(os.path.join(directory, 'train.out'), 'w') as f_out:
            for i, l in zip(range(train_size), np.random.normal(15, 3, train_size).astype(int)):
                l = max(l, 1)
                seq = np.random.choice(char_list, l)
                f_in.write(' '.join(seq) + '\n')
                f_out.write(' '.join(np.sort(seq)) + '\n')

        with open(os.path.join(directory, 'valid.in'), 'w') as f_in,\
            open(os.path.join(directory, 'valid.out'), 'w') as f_out:
            for i, l in zip(range(valid_size), np.random.normal(15, 3, valid_size).astype(int)):
                l = max(l, 1)
                seq = np.random.choice(char_list, l)
                f_in.write(' '.join(seq) + '\n')
                f_out.write(' '.join(np.sort(seq)) + '\n')

        with open(os.path.join(directory, 'test.in'), 'w') as f_in,\
            open(os.path.join(directory, 'test.out'), 'w') as f_out:
            for i, l in zip(range(test_size), np.random.normal(15, 3, test_size).astype(int)):
                l = max(l, 1)
                seq = np.random.choice(char_list, l)
                f_in.write(' '.join(seq) + '\n')
                f_out.write(' '.join(np.sort(seq)) + '\n')

        with open(os.path.join(directory, 'vocab.txt'), 'w') as f:
            for c in char_list:
                f.write(c + '\n')
