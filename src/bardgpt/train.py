from .config import *
from .model import BardGPT
from .data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm
import time
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

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
    optimizer = model.configure_optimizer(weight_decay=weight_decay, learning_rate=max_lr, device_type=device_type)
    linear = LinearLR(optimizer=optimizer, start_factor=1/warmup_steps, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer=optimizer, T_max=max_steps-warmup_steps, eta_min=min_lr)
    scheduler = SequentialLR(optimizer=optimizer, schedulers=[linear, cosine], milestones=[warmup_steps])
    train_loader = DataLoader(B=B, T=T, split='train', device=device_type)
    val_loader = DataLoader(B=B, T=T, split='val', device=device_type)
            
    enc = tiktoken.get_encoding('gpt2')

    max_length = 30
    num_return_sequences = 5

    prompt = enc.encode("Hello, I'm a language model,")
    prompt = torch.tensor(prompt, dtype=torch.long)
    prompt = prompt.to(device=device)
    prompt.unsqueeze_(dim=0)
    prompt = prompt.repeat(num_return_sequences, 1)

    val_loss = float('nan')
    pbar = tqdm(range(max_steps), desc='Training BardGPT', colour='#7BC621', dynamic_ncols=True)
    for step in pbar:
        last_step = (step == max_steps - 1)
        if (step > 0 and step % 100 == 0) or last_step:
            model.eval()
            val_loader.reset()
            with torch.inference_mode():
                val_steps = 20
                val_loss = 0.0
                for _ in range(val_steps):
                    x, y = val_loader.next_batch()
                    _, loss = model(x, y)
                    val_loss += loss.item() / val_steps

        if (step > 0 and step % 250 == 0) or last_step:
            model.eval()
            x_gen = prompt.clone()
            with torch.inference_mode():
                while x_gen.size(-1) < max_length:
                    logits, _ = model(x_gen)
                    logits = logits[:, -1, :]
                    probs = F.softmax(input=logits, dim=-1)
                    ix = torch.multinomial(input=probs, num_samples=1)
                    x_gen = torch.cat((x_gen, ix), dim=-1)
            print(f'\n{'=' * 24} GENERATION {'=' * 24}')
            for i in range(num_return_sequences):
                decoded = enc.decode(x_gen[i, :].tolist())
                print(f'\nSAMPLE {i+1} {colors.OKGREEN}{decoded}{colors.ENDC}')
            print(f'\n{'=' * 60}')

        model.train()
        t0 = time.time()
        optimizer.zero_grad()
        x, y = train_loader.next_batch()
        logits, loss = model(x, y)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        lr = scheduler.get_last_lr()[0]
        optimizer.step()
        scheduler.step()
        if device_type == 'cuda':
            torch.cuda.synchronize()
        elif device_type == 'mps':
            torch.mps.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T
        tok_sec = tokens_processed / dt
        pbar.set_postfix(train_loss=f'{loss.item():.6f}',
                         val_loss=f'{val_loss:.6f}',
                         norm=f'{norm:.4f}',
                         dt=f'{int(dt*1000)}ms',
                         lr=f'{lr:.6f}',
                         tok_sec=f'{tok_sec:.2f}')
