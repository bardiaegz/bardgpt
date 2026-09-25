from .config import *
from . import config as cfg
from .model import BardGPT
from .data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm
import time
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import dataclasses
import os
import argparse
import textwrap

class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.RawDescriptionHelpFormatter):
    """Keeps the description's line breaks AND appends each option's default."""


BANNER = r"""
  ____                _  _____ _____ _______ 
 |  _ \              | |/ ____|  __ \__   __|
 | |_) | __ _ _ __ __| | |  __| |__) | | |   
 |  _ < / _` | '__/ _` | | |_ |  ___/  | |   
 | |_) | (_| | | | (_| | |__| | |      | |   
 |____/ \__,_|_|  \__,_|\_____|_|      |_|   
"""

def main() -> None:

    parser = argparse.ArgumentParser(
        prog='uv run bardgpt',
        formatter_class=HelpFormatter,
        description=textwrap.dedent('''
        Train BardGPT from scratch in PyTorch.
        '''),
        epilog=textwrap.dedent('''
        Made with ♥︎  by Bardia Emamgholizadeh •︵•
        Linkedin: @bardiaegz | Telegram: @bardiaegz | Instagram: @bardia_egz
               ''')
    )

    parser.add_argument('--num-return-sequences', default=5, metavar='N', type=int, help=f'specify the number of sequences to generate in each {cfg.generation_eval_steps} steps')
    parser.add_argument('--norm', default='layer_norm', choices=('layer_norm', 'rmsnorm'), metavar='layer_norm|rmsnorm', help='choose how to normalize parameters')
    parser.add_argument('--activation', default='gelu', choices=('gelu', 'swiglu'), metavar='gelu|swiglu', help='choose what activation to use in the model')
    parser.add_argument('-B', '--batch-size', default=cfg.B, metavar='N', type=int, help='Choose batch size')
    parser.add_argument('-T', '--block-size', default=cfg.T, metavar='N', type=int, help='Choose block size')
    parser.add_argument('--max-steps', default=cfg.max_steps, metavar='N', type=int, help='Choose max steps. (If you want to choose another number, pay attention to --validation-eval-steps and --generation-eval-steps)')
    parser.add_argument('--validation-eval-steps', default=cfg.validation_eval_steps, metavar='N', type=int, help='Choose validation eval steps. (If you want to choose another number, pay attention to --max-steps and --generation-eval-steps)')
    parser.add_argument('--generation-eval-steps', default=cfg.generation_eval_steps, metavar='N', type=int, help='Choose generation eval steps. (If you want to choose another number, pay attention to --max-steps and --validation-eval-steps)')
    parser.add_argument('-s', '--seed', default=1337, type=int, metavar='N', help='Choose a number for seeding. Value must be within the inclusive range [-9,223,372,036,854,775,808 (-2⁶³), 18,446,744,073,709,551,615 (2⁶⁴-1)]. Negative inputs are remapped to positive values with the formula 18446744073709551615 + seed.')
    parser.add_argument('--lr', default=cfg.max_lr, type=float, metavar='N.n', help='Choose learning rate')
    parser.add_argument('--temperature', default=cfg.temperature, metavar='N.n', type=float, help='How much model creativity should be? I suggest not use number more than 2.')
    parser.add_argument('--top-k', default=cfg.k, type=int, metavar='N', help='Determine How many words does model should look at.')
    parser.add_argument('--top-p', default=cfg.p, type=float, metavar='N.n', help='Determine up to how much percentage does model should look at.')
    parser.add_argument('--resume', nargs='?', const='latest', default=None, metavar='PATH', help='resume from a checkpoint; bare --resume picks the newest in checkpoint/')
    args = parser.parse_args()

    num_return_sequences = args.num_return_sequences
    norm = args.norm
    activation = args.activation
    B = args.batch_size
    T = args.block_size
    assert args.max_steps >= max(args.validation_eval_steps, args.generation_eval_steps), f'max steps ({args.max_steps}) should be higher than validation ({args.validation_eval_steps}) and generation ({args.generation_eval_steps}) eval steps.' 
    max_steps = args.max_steps
    validation_eval_steps = args.validation_eval_steps
    generation_eval_steps = args.generation_eval_steps
    seed = args.seed
    assert args.lr > 0, 'Learning Rate must higher than 0.'
    max_lr = args.lr
    min_lr = 0.1 * max_lr  # GPT-3 App. B: cosine decays to 10% of peak
    assert args.temperature >= 0, 'Temperature can\'t be negative number'
    temperature = args.temperature
    assert 0 < args.top_k <= BardGPTConfig.vocab_size, f'K must be in (0, {BardGPTConfig.vocab_size}].'
    assert 0 < args.top_p <= 1, 'P must be in (0, 1].'
    k = args.top_k
    p = args.top_p


    print(BANNER)

    torch.manual_seed(seed=seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed(seed=seed)
    elif device_type == 'mps':
        torch.mps.manual_seed(seed=seed)

    log_file = open(file=os.path.join(log_dir, f'bardgpt-log-{time.strftime("%Y%m%d-%H%M%S", time.gmtime())}.txt'), mode='w')
    val_loss = float('nan')
    start_step = 0

    # resolve --resume into a checkpoint dict (None for a fresh run). the
    # checkpoint's own config wins, so the architecture always matches the
    # weights we are about to load.
    # reference: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
    ckpt = None
    if args.resume is not None:
        ckp_files = sorted(f for f in os.listdir(ckp_dir) if f.endswith('.pt'))
        assert ckp_files, f'no .pt checkpoints found in {ckp_dir}'
        if args.resume == 'latest':
            ckp_path = max((os.path.join(ckp_dir, f) for f in ckp_files), key=os.path.getmtime)
        else:
            assert args.resume in ckp_files, (
                f'no {args.resume} in {ckp_dir}. available: {", ".join(ckp_files)}')
            ckp_path = os.path.join(ckp_dir, args.resume)
        # load to cpu: portable across machines (a cuda checkpoint would
        # otherwise fail here on a non-cuda box) and avoids a transient 2x
        # spike on the accelerator. load_state_dict copies to the right device.
        ckpt = torch.load(f=ckp_path, map_location='cpu')
        model_config = BardGPTConfig(**ckpt['config'])
        for flag, saved in (('norm', model_config.norm), ('activation', model_config.activation)):
            given = getattr(args, flag)
            assert given == saved, (
                f'--{flag} {given} conflicts with the checkpoint ({saved}); '
                f'omit the flag or pass --{flag} {saved}')
        print(f'resuming from {os.path.basename(ckp_path)} '
              f'({model_config.norm}/{model_config.activation})')
    else:
        model_config = BardGPTConfig(norm=norm, activation=activation)

    # TensorFloat-32 (TF32) ->  Uses 1 sign bit, an 8-bit exponent (matching standard FP32), and a 10-bit mantissa (matching FP16/half precision) for a total of 19 bits.
    # https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/
    # https://developer.nvidia.com/blog/getting-immediate-speedups-with-a100-tf32/
    torch.set_float32_matmul_precision('high')
    model = BardGPT(model_config)
    model.to(device=device)
    optimizer = model.configure_optimizer(weight_decay=weight_decay, learning_rate=max_lr, device_type=device_type)
    linear = LinearLR(optimizer=optimizer, start_factor=1/warmup_steps, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer=optimizer, T_max=max_steps-warmup_steps, eta_min=min_lr)
    scheduler = SequentialLR(optimizer=optimizer, schedulers=[linear, cosine], milestones=[warmup_steps])

    train_loader = DataLoader(B=B, T=T, split='train', device=device_type)
    val_loader = DataLoader(B=B, T=T, split='val', device=device_type)

    # restore everything else AFTER the generators are seeded above, otherwise
    # manual_seed would clobber the checkpoint's rng state.
    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        train_loader.current_position = ckpt['train_position']
        torch.set_rng_state(ckpt['rng_cpu'])
        if ckpt['rng_device'] is not None:
            if device_type == 'cuda':
                torch.cuda.set_rng_state_all(ckpt['rng_device'])
            elif device_type == 'mps':
                torch.mps.set_rng_state(ckpt['rng_device'])
        val_loss = ckpt['val_loss']
        start_step = ckpt['step'] + 1   # ckpt['step'] already finished
        assert start_step < max_steps, (
            f'checkpoint is at step {ckpt["step"]}, which is already past '
            f'--max-steps {max_steps}; raise --max-steps to continue')
            
    enc = tiktoken.get_encoding('gpt2')

    prompt = enc.encode("Hello, I'm a language model,")
    prompt = torch.tensor(prompt, dtype=torch.long)
    prompt = prompt.to(device=device)
    prompt.unsqueeze_(dim=0)
    prompt = prompt.repeat(num_return_sequences, 1)


    pbar = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc='Training BardGPT', colour='#7BC621', dynamic_ncols=True)
    for step in pbar:
        last_step = (step == max_steps - 1)
        if (step > 0 and step % validation_eval_steps == 0) or last_step:
            model.eval()
            val_loader.reset()
            with torch.inference_mode():
                val_steps = 20
                val_loss = 0.0
                for _ in range(val_steps):
                    x, y = val_loader.next_batch()
                    # BrainFloat-16 (BF16) -> Uses 1 sign bit, an 8-bit exponent (matching standard FP32), and a 7-bit mantissa for a total of 16 bits.
                    # https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
                    # https://www.cerebras.ai/blog/to-bfloat-or-not-to-bfloat-that-is-the-question
                    # https://docs.cloud.google.com/tpu/docs/bfloat16
                    # https://arxiv.org/pdf/1905.12322
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        _, loss = model(x, y)
                    val_loss += loss.item() / val_steps
                log_file.write(f'\nSTEP {step:05d} | VAL LOSS: {val_loss:.6f}')

        if (step > 0 and step % generation_eval_steps == 0) or last_step:
            model.eval()
            x_gen = prompt.clone()
            with torch.inference_mode():
                while x_gen.size(-1) < max_length:
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, _ = model(x_gen)
                    logits = logits[:, -1, :]
                    if temperature == 0:
                        xcol = logits.argmax(dim=-1, keepdim=True)
                    else:
                        logits = logits / temperaturelfmode
                        probs = F.softmax(input=logits, dim=-1)
                        topk_probs, topk_indices = torch.topk(input=probs, k=k, dim=-1)

                        # top-p (nucleus): keep the smallest prefix whose mass reaches p.
                        # torch.topk already returns descending order, so cumsum is the
                        # running mass. shift the mask right by one so the token that
                        # crosses p is KEPT, otherwise a single token with prob > p
                        # would mask the whole row. https://arxiv.org/abs/1904.09751
                        cumsum = torch.cumsum(input=topk_probs, dim=-1)
                        remove = cumsum - topk_probs > p
                        topk_probs = topk_probs.masked_fill(mask=remove, value=0.0)
                        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

                        ix = torch.multinomial(input=topk_probs, num_samples=1)
                        xcol = torch.gather(input=topk_indices, dim=-1, index=ix)
                    x_gen = torch.cat((x_gen, xcol), dim=-1)

            log_file.write(f'\n{'=' * 24} GENERATION {'=' * 24}')
            print(f'\n{'=' * 24} GENERATION {'=' * 24}')
            for i in range(num_return_sequences):
                decoded = enc.decode(x_gen[i, :].tolist())
                log_file.write(f'\nSAMPLE {i+1} -> {decoded}')
                print(f'\nSAMPLE {i+1} {colors.OKGREEN}{decoded}{colors.ENDC}')
            log_file.write(f'\n{'=' * 60}')
            print(f'\n{'=' * 60}')

        model.train()
        t0 = time.time()
        optimizer.zero_grad()
        x, y = train_loader.next_batch()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        # Appendix B on GPT-3 paper
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
        log_file.write(f'\nSTEP {step:05d} | TRAIN LOSS: {loss.item():.6f} | NORM: {norm:.4f} | DT: {int(dt*1000)}ms | LR: {lr:.6f} | TOK/SEC: {tok_sec:.2f}')
        pbar.set_postfix(train_loss=f'{loss.item():.6f}',
                         val_loss=f'{val_loss:.6f}',
                         norm=f'{norm:.4f}',
                         dt=f'{int(dt*1000)}ms',
                         lr=f'{lr:.6f}',
                         tok_sec=f'{tok_sec:.2f}')
        if (step > 0 and step % validation_eval_steps == 0) or last_step:
            checkpoint = {
                'config': dataclasses.asdict(model.config),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'step': step,
                'train_position': train_loader.current_position,
                'rng_cpu': torch.get_rng_state(),
                'rng_device': torch.mps.get_rng_state() if device_type == 'mps' else (torch.cuda.get_rng_state_all() if device_type == 'cuda' else None),
                'val_loss': val_loss,
            }
            filename = os.path.join(ckp_dir, f'bardgpt-{step:05d}.pt')
            torch.save(obj=checkpoint, f=filename)
            print(f'\ncheckpoint saved to: {filename}')
        log_file.flush()
    log_file.close()
