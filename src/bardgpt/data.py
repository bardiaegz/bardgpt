from .config import *
from functools import cache
import torch
import tiktoken
import os
import numpy as np

DATASETS = ('edufineweb', 'tinyshakespeare')

input_file = os.path.join(proj_dir, 'input.txt')
shard_dir = os.path.join(proj_dir, 'edu_fineweb10B')


@cache
def load_shakespeare():
    """Read + BPE-encode input.txt once; both splits are views on one tensor."""
    with open(file=input_file, mode='r') as f:
        text = f.read()
    enc = tiktoken.get_encoding('gpt2')
    return torch.tensor(data=enc.encode(text=text), dtype=torch.long)


def load_tokens(filename):
    """Load one uint16 shard. int32 first because torch has no uint16."""
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


class DataLoader:
    """Serves (x, y) batches from either backend.

    tinyshakespeare is a single in-memory tensor split 90/10. edufineweb is a
    directory of 100M-token shards loaded one at a time, advancing round-robin.
    Both expose current_shard / current_position so a checkpoint can resume
    mid-epoch; tinyshakespeare simply always reports shard 0.
    """

    def __init__(self, B, T, split='train', device='cpu', dataset='edufineweb'):
        assert split in {'train', 'val'}
        assert dataset in DATASETS, f'dataset must be one of {DATASETS}'
        self.B, self.T, self.split, self.device, self.dataset = B, T, split, device, dataset

        if dataset == 'tinyshakespeare':
            assert os.path.exists(input_file), f'{input_file} not found'
            n = 0.9
            tokens = load_shakespeare()
            cut = int(n * len(tokens))
            self.splits = [tokens[:cut] if split == 'train' else tokens[cut:]]
            print(f'loaded {len(self.splits[0]):,} tokens for split {split} (tinyshakespeare)')
        else:
            assert os.path.isdir(shard_dir), (
                f'{shard_dir} not found. run: uv run bardgpt-data '
                f'(or use --dataset tinyshakespeare)')
            shards = sorted(s for s in os.listdir(shard_dir) if split in s and s.endswith('.npy'))
            assert shards, f'no {split} shards in {shard_dir}'
            self.splits = [os.path.join(shard_dir, s) for s in shards]
            print(f'found {len(self.splits)} shards for split {split} (edufineweb)')

        self.reset()

    def _load(self, i):
        s = self.splits[i]
        return s if self.dataset == 'tinyshakespeare' else load_tokens(s)

    def reset(self):
        self.current_shard = 0
        self.tokens = self._load(0)
        self.current_position = 0

    def set_state(self, shard, position):
        """Restore a checkpointed position, reloading the shard if needed."""
        assert 0 <= shard < len(self.splits), (
            f'checkpoint refers to shard {shard} but only {len(self.splits)} exist')
        if shard != self.current_shard:
            self.current_shard = shard
            self.tokens = self._load(shard)
        self.current_position = position

    def next_batch(self):
        B, T, device = self.B, self.T, self.device

        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        buf = buf.to(device=device)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T

        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.splits)
            self.tokens = self._load(self.current_shard)
            self.current_position = 0

        return x, y
