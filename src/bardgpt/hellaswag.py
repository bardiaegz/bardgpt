"""HellaSwag evaluation.

HellaSwag (Zellers et al. 2019, https://arxiv.org/abs/1905.07830) gives a
context and four candidate endings, one of which really happened. The wrong
ones are adversarially generated: grammatical, but nonsense.

A base LM cannot be *asked* a multiple-choice question, so instead we score how
surprised the model is by each ending and pick the least surprising one. Two
numbers come out of that:

    acc      - lowest summed completion loss wins
    acc_norm - lowest MEAN per-token loss wins (length-normalised)

acc_norm is the headline metric everyone reports, because a raw sum
systematically favours shorter endings. For calibration: random is 25%,
GPT-2 124M scores ~29.5%, GPT-2 1.5B ~50%, humans ~95%. A 124M model landing
just above chance is the correct result, not a bug.
"""

from .config import *
from .ddp import *
from .model import BardGPT
import torch
import torch.nn.functional as F
import tiktoken
import argparse
import os
import textwrap

enc = tiktoken.get_encoding('gpt2')


def render_example(example):
    """One example -> (tokens, mask, label).

    tokens (4, N) is each ending appended to the shared context, right-padded.
    mask   (4, N) is 1 on completion tokens, 0 on context and padding -- only
                  the completion is scored, since the context is identical
                  across all four rows and would just add a constant.
    """
    ctx_tokens = enc.encode(example['ctx'])
    rows, masks = [], []
    for ending in example['endings']:
        # leading space: endings continue the sentence, and GPT-2 BPE encodes
        # ' word' differently from 'word'
        end_tokens = enc.encode(' ' + ending)
        rows.append(ctx_tokens + end_tokens)
        masks.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    n = max(len(r) for r in rows)
    tokens = torch.zeros((4, n), dtype=torch.long)
    mask = torch.zeros((4, n), dtype=torch.long)
    for i, (r, m) in enumerate(zip(rows, masks)):
        tokens[i, :len(r)] = torch.tensor(r)
        mask[i, :len(m)] = torch.tensor(m)

    return tokens, mask, int(example['label'])


@torch.inference_mode()
def score_example(model, tokens, mask):
    """Return (sum_loss, mean_loss) per row -- lower is better."""
    tokens, mask = tokens.to(device), mask.to(device)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits, _ = model(tokens)

    # logits at position i predict token i+1, so both the targets and the mask
    # shift left by one. getting this off by one silently pins you at 25%.
    shift_logits = logits[:, :-1, :].float()
    shift_tokens = tokens[:, 1:]
    shift_mask = mask[:, 1:]

    losses = F.cross_entropy(
        input=shift_logits.reshape(-1, shift_logits.size(-1)),
        target=shift_tokens.reshape(-1),
        reduction='none',
    ).view(shift_tokens.size())

    losses = losses * shift_mask          # drop context and padding
    sum_loss = losses.sum(dim=1)
    mean_loss = sum_loss / shift_mask.sum(dim=1)
    return sum_loss, mean_loss


def iter_examples(split='validation'):
    from datasets import load_dataset
    return load_dataset('Rowan/hellaswag', split=split)


def evaluate(model, limit=None, split='validation', progress=False):
    """Score the model on HellaSwag. Returns (acc, acc_norm, n).

    Under DDP each rank takes every world_size-th example and the counts are
    summed at the end, so every rank returns the same figures.
    """
    model.eval()
    n_total = n_correct = n_correct_norm = 0

    examples = iter_examples(split)
    if limit is not None:
        examples = examples.select(range(min(limit, len(examples))))

    it = enumerate(examples)
    if progress and master_process:
        from tqdm import tqdm
        it = tqdm(it, total=len(examples), desc='HellaSwag', colour='#7BC621')

    for i, example in it:
        if i % ddp_world_size != ddp_rank:      # shard across ranks
            continue
        tokens, mask, label = render_example(example)
        sum_loss, mean_loss = score_example(model, tokens, mask)
        n_total += 1
        n_correct += int(sum_loss.argmin().item() == label)
        n_correct_norm += int(mean_loss.argmin().item() == label)

    if ddp:
        import torch.distributed as dist
        counts = torch.tensor([n_total, n_correct, n_correct_norm],
                              dtype=torch.long, device=device)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        n_total, n_correct, n_correct_norm = counts.tolist()

    return n_correct / n_total, n_correct_norm / n_total, n_total


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='uv run bardgpt-hellaswag',
        formatter_class=HelpFormatter,
        description=textwrap.dedent('''
        Evaluate a BardGPT checkpoint on HellaSwag.

        Scores each of the four candidate endings by completion loss and picks
        the lowest. Random is 25%; GPT-2 124M scores about 29.5% acc_norm.
        '''),
        epilog=textwrap.dedent('''
        Made with ♥︎  by Bardia Emamgholizadeh •︵•
        Linkedin: @bardiaegz | Telegram: @bardiaegz | Instagram: @bardia_egz
               ''')
    )
    parser.add_argument('-c', '--checkpoint', default='latest', metavar='PATH', help="checkpoint in checkpoint/ to evaluate; 'latest' picks the newest")
    parser.add_argument('-l', '--limit', default=None, metavar='N', type=int, help='only score the first N examples (default: all 10,042)')
    parser.add_argument('--split', default='validation', choices=('validation', 'train'), metavar='validation|train', help='which split to score')
    args = parser.parse_args()

    from .generate import resolve_checkpoint

    print(BANNER)
    ckp_path = resolve_checkpoint(args.checkpoint)
    ckpt = torch.load(f=ckp_path, map_location='cpu')
    model_config = BardGPTConfig(**ckpt['config'])

    model = BardGPT(model_config)
    model.load_state_dict(ckpt['model'])
    model.to(device=device)

    print(f'checkpoint : {os.path.basename(ckp_path)}')
    print(f'trained to : step {ckpt["step"]}, val loss {ckpt.get("val_loss", float("nan")):.4f}')

    acc, acc_norm, n = evaluate(model, limit=args.limit, split=args.split, progress=True)

    print(f'\n{"=" * 24} HELLASWAG {"=" * 25}')
    print(f'\nexamples  : {n:,}')
    print(f'acc       : {colors.OKGREEN}{acc * 100:.2f}%{colors.ENDC}')
    print(f'acc_norm  : {colors.OKGREEN}{acc_norm * 100:.2f}%{colors.ENDC}   (random 25.00%, GPT-2 124M ~29.5%)')
    print(f'\n{"=" * 60}')


if __name__ == '__main__':
    main()
