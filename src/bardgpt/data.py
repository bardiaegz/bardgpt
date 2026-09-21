from .config import *
from functools import cache
import torch
import tiktoken
import os

input_file = os.path.join(proj_dir, 'input.txt')

@cache
def load_tokens():
    with open(file=input_file, mode='r') as f:
        text = f.read()
    enc = tiktoken.get_encoding('gpt2')
    return torch.tensor(data=enc.encode(text=text), dtype=torch.long)

class DataLoader:
    def __init__(self, B, T, split='train', device='cpu'):
        assert split in {'train', 'val'}
        self.B, self.T, self.split, self.device = B, T, split, device
        n = 0.9

        tokens = load_tokens()
        self.tokens = tokens[:int(n * len(tokens))] if split == 'train' else tokens[int(n * len(tokens)):]

        print(f'loaded {len(self.tokens)} tokens for split {split}')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches.')

        self.reset()

    def reset(self):
        self.current_position = 0

    def next_batch(self):
        B, T, device = self.B, self.T, self.device

        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        buf = buf.to(device=device)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T

        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y