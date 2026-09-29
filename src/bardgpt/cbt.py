"""Children's Book Test (CBT) evaluation.

CBT (Hill et al. 2016, https://arxiv.org/abs/1511.02301) gives 20 sentences of
context plus a 21st query sentence with one word replaced by XXXXX, and ten
candidate words. Pick the right one.

Two subsets are usually reported:

    CN - common nouns     (man, place, sight, ...)
    NE - named entities   (Alice, London, ...)

Scoring follows the GPT-2 paper section 3.2: substitute each candidate into the
blank, then score the candidate *and the rest of the query sentence* conditioned
on everything before it. Highest total probability wins. Scoring the remainder
matters -- a wrong noun often only becomes implausible once you see what follows.

The raw dataset is PTB-tokenised (-LRB-, ` ` quotes, spaces before punctuation).
GPT-2 never saw text like that, so we detokenise first; the paper does the same
with what it calls invertible de-tokenizers. Skipping this costs several points.

GPT-2 targets (paper, Table 3, accuracy):
    CBT-CN  124M 87.65 | 345M 92.35 | 762M 93.45 | 1.5B 93.30
    CBT-NE  124M 83.40 | 345M 87.10 | 762M 88.00 | 1.5B 89.05
Random over ten options is 10%.
"""

from .config import *
from .ddp import *
from .model import BardGPT
import torch
import torch.nn.functional as F
import tiktoken
import argparse
import os
import re
import textwrap

enc = tiktoken.get_encoding('gpt2')

SUBSETS = ('CN', 'NE', 'V', 'P')

_BRACKETS = {'-LRB-': '(', '-RRB-': ')', '-LSB-': '[', '-RSB-': ']',
             '-LCB-': '{', '-RCB-': '}'}


def detokenize(text):
    """Undo PTB tokenisation so the text looks like what GPT-2 was trained on."""
    for tag, ch in _BRACKETS.items():
        text = text.replace(tag, ch)
    text = text.replace('``', '"').replace("''", '"')
    text = re.sub(r"\s*`\s*", " '", text)          # ` quoted ' -> ' quoted '
    text = re.sub(r"\s+([,.!?;:%])", r'\1', text)  # no space before punctuation
    text = re.sub(r"\s+('s|'re|'ve|'ll|'d|'m|n't)\b", r'\1', text)
    text = re.sub(r"\(\s+", '(', text)
    text = re.sub(r"\s+\)", ')', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def render_example(example):
    """One example -> (rows, label).

    rows is a list of (tokens, n_scored) -- one per candidate. n_scored covers
    the candidate plus the remainder of the query sentence.
    """
    context = detokenize(' '.join(example['sentences']))
    question = detokenize(example['question'])

    # split on the blank. keeping the space with the FOLLOWING text matches
    # GPT-2 BPE, which encodes ' word' as one token.
    before, after = question.split('XXXXX', 1)
    before = before.rstrip()
    after = after.rstrip()

    prefix_tokens = enc.encode(context + ' ' + before)
    label = example['options'].index(example['answer'])

    rows = []
    for option in example['options']:
        tail_tokens = enc.encode(' ' + option + after)
        rows.append((prefix_tokens + tail_tokens, len(tail_tokens)))
    return rows, label


@torch.inference_mode()
def score_rows(model, rows):
    """Return the summed loss of the scored span for each candidate."""
    block = model.config.block_size
    scores = []
    for tokens, n_scored in rows:
        tokens = tokens[-block:]                       # keep the tail if too long
        t = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = model(t)

        # logits at position i predict token i+1
        shift_logits = logits[:, :-1, :].float()[:, -n_scored:, :]
        shift_tokens = t[:, 1:][:, -n_scored:]
        loss = F.cross_entropy(
            input=shift_logits.reshape(-1, shift_logits.size(-1)),
            target=shift_tokens.reshape(-1),
            reduction='sum',
        )
        scores.append(loss)
    return torch.stack(scores)


def iter_examples(subset='CN', split='test'):
    from datasets import load_dataset
    return load_dataset('cam-cst/cbt', subset, split=split)


def evaluate(model, subset='CN', limit=None, split='test', progress=False):
    """Score the model on CBT. Returns (acc, acc_norm, n).

    acc      - lowest total loss over the scored span wins
    acc_norm - the same, divided by span length

    The spans differ in length only through the candidate itself (usually one
    or two tokens), so the two numbers track each other closely -- unlike
    HellaSwag, where endings differ wildly in length.
    """
    model.eval()
    n_total = n_correct = n_correct_norm = 0

    examples = iter_examples(subset, split)
    if limit is not None:
        examples = examples.select(range(min(limit, len(examples))))

    it = enumerate(examples)
    if progress and master_process:
        from tqdm import tqdm
        it = tqdm(it, total=len(examples), desc=f'CBT-{subset}', colour='#7BC621')

    for i, example in it:
        if i % ddp_world_size != ddp_rank:      # shard across ranks
            continue
        rows, label = render_example(example)
        losses = score_rows(model, rows)
        lengths = torch.tensor([n for _, n in rows], dtype=torch.float, device=device)
        n_total += 1
        n_correct += int(losses.argmin().item() == label)
        n_correct_norm += int((losses / lengths).argmin().item() == label)

    if ddp:
        import torch.distributed as dist
        counts = torch.tensor([n_total, n_correct, n_correct_norm],
                              dtype=torch.long, device=device)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        n_total, n_correct, n_correct_norm = counts.tolist()

    return n_correct / n_total, n_correct_norm / n_total, n_total


GPT2_124M = {'CN': 87.65, 'NE': 83.40}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='uv run bardgpt-cbt',
        formatter_class=HelpFormatter,
        description=textwrap.dedent('''
        Evaluate a BardGPT checkpoint on the Children's Book Test.

        Substitutes each of ten candidates into the blank and scores the
        candidate plus the rest of the sentence. Random is 10%;
        GPT-2 124M scores 87.65 on CN and 83.40 on NE.
        '''),
        epilog=textwrap.dedent('''
        Made with ♥︎  by Bardia Emamgholizadeh •︵•
        Linkedin: @bardiaegz | Telegram: @bardiaegz | Instagram: @bardia_egz
               ''')
    )
    parser.add_argument('-c', '--checkpoint', default='latest', metavar='PATH', help="checkpoint in checkpoint/ to evaluate; 'latest' picks the newest")
    parser.add_argument('--subset', default='CN', choices=SUBSETS, metavar='CN|NE|V|P', help='CN common nouns, NE named entities, V verbs, P prepositions')
    parser.add_argument('-l', '--limit', default=None, metavar='N', type=int, help='only score the first N examples (test split has 2,500)')
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

    acc, acc_norm, n = evaluate(model, subset=args.subset, limit=args.limit, progress=True)

    target = GPT2_124M.get(args.subset)
    hint = f'   (GPT-2 124M {target:.2f}%)' if target else ''
    print(f'\n{"=" * 25} CBT-{args.subset} {"=" * (27 - len(args.subset))}')
    print(f'\nexamples  : {n:,}')
    print(f'acc       : {colors.OKGREEN}{acc * 100:.2f}%{colors.ENDC}{hint}')
    print(f'acc_norm  : {colors.OKGREEN}{acc_norm * 100:.2f}%{colors.ENDC}   (random 10.00%)')
    print(f'\n{"=" * 60}')


if __name__ == '__main__':
    main()
