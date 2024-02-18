import os
import sys
import zipfile
import numpy as np
import pickle

if not os.path.exists('data/enwik_char/enwik8.zip'):
    print("enwik8.zip file not found.")
    sys.exit()

# Check if processed files already exist
if os.path.exists('data/enwik_char/train.bin') and os.path.exists('data/enwik_char/valid.bin') and os.path.exists('data/enwik_char/test.bin'):
    print('Processed files already exist - skipping processing')
    sys.exit()

with zipfile.ZipFile('data/enwik_char/enwik8.zip') as zf:
    # Assume the file inside is named 'enwik8', replace if it's different
    data_bytes = zf.read('enwik8')
    data = data_bytes.decode('utf-8')  # Decode the bytes to a string

print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):int(n*0.95)]
test_data = data[int(n*0.95):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"test has {len(test_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
test_ids.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)