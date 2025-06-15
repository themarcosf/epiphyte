import os
import pickle
import random

import torch
from transformers import GPT2Tokenizer

local_dir = 'seed_encodings'

# create the local directory for seed encodings
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# get the vocabulary
vocab = tokenizer.get_vocab()

# create a random array of tokens
unique_randoms = random.sample(range(len(vocab)), len(vocab))

# create a randomized vocabulary mapping
vocab_randomized = {v: r for v, r in zip(vocab.values(), unique_randoms)}
vocab_randomized_tensor = torch.tensor([vocab_randomized[i] for i in range(len(vocab_randomized))])

# save the randomized vocabulary mapping
filepath = os.path.join(DATA_CACHE_DIR, 'gpt2_seed_encodings.pkl')

with open(filepath, 'wb') as f:
    pickle.dump(vocab_randomized_tensor, f)