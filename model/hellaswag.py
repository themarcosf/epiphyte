"""
HellaSwag: Can a Machine Really Finish Your Sentence?
Authors: Zellers R., et. al.
https://arxiv.org/pdf/1905.07830
https://github.com/rowanz/hellaswag

GPT2 124M observed accuracy (norm): 0.2957 <<-- this is the baseline
"""
import json
import os
import pickle
import requests

import tiktoken
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import GPT2LMHeadModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CACHE_DIR = os.path.join(BASE_DIR, 'hellaswag')
ENCODINGS_CACHE_DIR = os.path.join(BASE_DIR, 'seed_encodings')


enc = tiktoken.get_encoding('gpt2')

seeds = os.path.join(ENCODINGS_CACHE_DIR, 'gpt2_seed_encodings.pkl')
with open(seeds, 'rb') as f:
    seed_encodings: torch.Tensor = pickle.load(f)


def download_file(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    'train': 'https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl',
    'val': 'https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl',
    'test': 'https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl',
}

def get_seed_encodings(ctx):
    tkns = enc.encode(ctx)
    seed_tkns = seed_encodings[tkns].tolist()
    return seed_tkns

def decode_seed_encodings(seed_tkns):
    mask = torch.isin(seed_encodings, torch.tensor(seed_tkns))
    tkns = mask.nonzero(as_tuple=True)[0]
    return enc.decode(tkns.tolist())


def download(split):
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f'hellaswag_{split}.jsonl')
    if not os.path.exists(data_filename):
        print(f'Downloading {data_url} to {data_filename}...')
        download_file(data_url, data_filename)


def render_example(example):
    ctx = example['ctx']
    label = example['label']
    endings = example['endings']

    data = {
        'label': label,
        'ctx_tokens': None,
        'ending_tokens': [],
    }

    ctx_tokens = get_seed_encodings(ctx)
    data['ctx_tokens'] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = get_seed_encodings(' ' + end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data['ending_tokens'].append(end_tokens)

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def iterate_examples(split):
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f'hellaswag_{split}.jsonl'), 'r') as f:
        for line in f:
            example = json.loads(line)
            yield example

@torch.no_grad()
def evaluate(model_type, device):

    torch.set_float32_matmul_precision('high') # use tf32
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    model = torch.compile(model)

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples('val'):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        logits = model(tokens).logits
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
    
        shift_mask = (mask[..., 1:]).contiguous()
        masked_shift_losses = shift_losses * shift_mask
        
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f'{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}')

        if num_total < 10:
            print('---')
            print(f'Context:\n {example['ctx']}')
            print(f'Endings:')
            for i, end in enumerate(example['endings']):
                print(f'{i} (loss: {avg_loss[i].item():.4f}) {end}')
            print(f'predicted: {pred_norm}, actual: {label}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_type', type=str, default='gpt2', help='the model type to use')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='the device to use')
    args = parser.parse_args()
    evaluate(args.model_type, args.device)