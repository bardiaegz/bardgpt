from .config import *
from .model import BardGPT
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

BANNER = r"""
  ____                _  _____ _____ _______ 
 |  _ \              | |/ ____|  __ \__   __|
 | |_) | __ _ _ __ __| | |  __| |__) | | |   
 |  _ < / _` | '__/ _` | | |_ |  ___/  | |   
 | |_) | (_| | | | (_| | |__| | |      | |   
 |____/ \__,_|_|  \__,_|\_____|_|      |_|   
"""

def main() -> None:
    print(BANNER)
    model = BardGPT(BardGPTConfig())
    model.to(device=device)

    enc = tiktoken.get_encoding('gpt2')

    max_length = 30
    num_return_sequences = 5

    prompt = enc.encode("Hello, I'm a language model,")
    prompt = torch.tensor(prompt, dtype=torch.long)
    prompt = prompt.to(device=device)
    prompt.unsqueeze_(dim=0)
    prompt = prompt.repeat(num_return_sequences, 1)

    with torch.inference_mode():
        while prompt.size(-1) < max_length:
            logits, _ = model(prompt)
            logits = logits[:, -1, :]
            probs = F.softmax(input=logits, dim=-1)
            ix = torch.multinomial(input=probs, num_samples=1)
            prompt = torch.cat((prompt, ix), dim=-1)
    print(f'\n{'=' * 24} GENERATION {'=' * 24}')
    for i in range(num_return_sequences):
        decoded = enc.decode(prompt[i, :].tolist())
        print(f'\nSAMPLE {i+1} {colors.OKGREEN}{decoded}{colors.ENDC}')
    print(f'\n{'=' * 60}')