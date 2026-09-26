from .config import *
from . import config as cfg
from .model import BardGPT
import torch
import torch.nn.functional as F
import tiktoken
import argparse
import os
import textwrap


def resolve_checkpoint(name):
    """Turn a --checkpoint value into a path. 'latest' picks the newest by mtime."""
    files = sorted(f for f in os.listdir(ckp_dir) if f.endswith('.pt'))
    assert files, f'no .pt checkpoints found in {ckp_dir}'
    if name == 'latest':
        return max((os.path.join(ckp_dir, f) for f in files), key=os.path.getmtime)
    assert name in files, f'no {name} in {ckp_dir}. available: {", ".join(files)}'
    return os.path.join(ckp_dir, name)


@torch.inference_mode()
def generate(model, idx, max_length, temperature, k, p):
    """Autoregressively extend idx (B, T) until it reaches max_length tokens.

    temperature == 0 is greedy. otherwise logits are scaled, truncated to the
    top k, then to the smallest prefix whose mass reaches p, then sampled.
    """
    while idx.size(-1) < max_length:
        # only the last block_size tokens fit in the context window
        window = idx[:, -model.config.block_size:]
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = model(window)
        logits = logits[:, -1, :]

        if temperature == 0:
            xcol = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            probs = F.softmax(input=logits, dim=-1)
            topk_probs, topk_indices = torch.topk(input=probs, k=k, dim=-1)

            # top-p (nucleus): keep the smallest prefix whose mass reaches p.
            # torch.topk already returns descending order, so cumsum is the
            # running mass. subtracting each token's own probability gives the
            # mass BEFORE it, so the token that crosses p is KEPT -- otherwise a
            # single token with prob > p would mask the whole row.
            # https://arxiv.org/abs/1904.09751
            cumsum = torch.cumsum(input=topk_probs, dim=-1)
            remove = cumsum - topk_probs > p
            topk_probs = topk_probs.masked_fill(mask=remove, value=0.0)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

            ix = torch.multinomial(input=topk_probs, num_samples=1)
            xcol = torch.gather(input=topk_indices, dim=-1, index=ix)

        idx = torch.cat((idx, xcol), dim=-1)
    return idx


def main() -> None:

    parser = argparse.ArgumentParser(
        prog='uv run bardgpt-generate',
        formatter_class=HelpFormatter,
        description=textwrap.dedent('''
        Sample text from a trained BardGPT checkpoint.
        '''),
        epilog=textwrap.dedent('''
        Made with ♥︎  by Bardia Emamgholizadeh •︵•
        Linkedin: @bardiaegz | Telegram: @bardiaegz | Instagram: @bardia_egz
               ''')
    )

    parser.add_argument('-c', '--checkpoint', default='latest', metavar='PATH', help="checkpoint in checkpoint/ to sample from; 'latest' picks the newest")
    parser.add_argument('--prompt', default="Hello, I'm a language model,", metavar='TEXT', help='text to continue')
    parser.add_argument('-n', '--num-return-sequences', default=cfg.num_return_sequences, metavar='N', type=int, help='how many continuations to generate')
    parser.add_argument('--max-length', default=cfg.max_length, metavar='N', type=int, help='total sequence length, prompt included')
    parser.add_argument('--temperature', default=cfg.temperature, metavar='N.n', type=float, help='0 is greedy; higher is more random')
    parser.add_argument('--top-k', default=cfg.k, metavar='N', type=int, help='keep only the k most likely tokens')
    parser.add_argument('--top-p', default=cfg.p, metavar='N.n', type=float, help='keep the smallest set of tokens whose mass reaches p')
    parser.add_argument('-s', '--seed', default=1337, metavar='N', type=int, help='RNG seed, for reproducible samples')
    args = parser.parse_args()

    assert args.num_return_sequences > 0, 'num return sequences must be higher than 0.'
    assert args.max_length > 0, 'max length must be higher than 0.'
    assert args.temperature >= 0, "temperature can't be a negative number"
    assert 0 < args.top_k <= BardGPTConfig.vocab_size, f'K must be in (0, {BardGPTConfig.vocab_size}].'
    assert 0 < args.top_p <= 1, 'P must be in (0, 1].'

    print(BANNER)

    torch.manual_seed(seed=args.seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed(seed=args.seed)
    elif device_type == 'mps':
        torch.mps.manual_seed(seed=args.seed)

    ckp_path = resolve_checkpoint(args.checkpoint)
    ckpt = torch.load(f=ckp_path, map_location='cpu')
    model_config = BardGPTConfig(**ckpt['config'])

    model = BardGPT(model_config)
    model.load_state_dict(ckpt['model'])
    model.to(device=device)
    model.eval()

    val_loss = ckpt.get('val_loss', float('nan'))
    print(f'checkpoint : {os.path.basename(ckp_path)}')
    print(f'trained to : step {ckpt["step"]}, val loss {val_loss:.4f}')
    print(f'architecture: {model_config.norm} / {model_config.activation}, '
          f'{sum(p.numel() for p in model.parameters()):,} params')

    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(args.prompt)
    assert len(tokens) < args.max_length, (
        f'prompt is {len(tokens)} tokens but --max-length is {args.max_length}; '
        f'nothing left to generate')

    idx = torch.tensor(tokens, dtype=torch.long, device=device)
    idx = idx.unsqueeze(dim=0).repeat(args.num_return_sequences, 1)

    idx = generate(model, idx, args.max_length, args.temperature, args.top_k, args.top_p)

    print(f'\n{"=" * 24} GENERATION {"=" * 24}')
    for i in range(args.num_return_sequences):
        decoded = enc.decode(idx[i, :].tolist())
        print(f'\nSAMPLE {i+1} {colors.OKGREEN}{decoded}{colors.ENDC}')
    print(f'\n{"=" * 60}')


if __name__ == '__main__':
    main()
