from .config import proj_dir

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

DATA_CACHE_DIR = os.path.join(proj_dir, 'edu_fineweb10B')
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

shard_size = int(1e8)

enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens['<|endoftext|>']


def tokenize(doc):
    """One document -> uint16 token array, prefixed with the EOT delimiter."""
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc['text']))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), 'token dictionary too large for uint16'
    return tokens_np.astype(np.uint16)


def write_datafile(filename, tokens_np):
    np.save(file=filename, arr=tokens_np)


def main() -> None:
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    nprocs = max(1, os.cpu_count() // 2)
    print(f'tokenizing fineweb-edu sample-10BT with {nprocs} processes')
    print(f'shards of {shard_size:,} tokens -> {DATA_CACHE_DIR}')

    with mp.Pool(processes=nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(func=tokenize, iterable=fw, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit='tokens', desc=f'Shard {shard_index}')
                progress_bar.update(len(tokens))
            else:
                split = 'val' if shard_index == 0 else 'train'
                filename = os.path.join(DATA_CACHE_DIR, f'edufineweb_{split}_{shard_index:06d}')
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        if token_count != 0:
            split = 'val' if shard_index == 0 else 'train'
            filename = os.path.join(DATA_CACHE_DIR, f'edufineweb_{split}_{shard_index:06d}')
            write_datafile(filename, all_tokens_np[:token_count])


if __name__ == '__main__':
    main()
