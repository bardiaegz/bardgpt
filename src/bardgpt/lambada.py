"""LAMBADA evaluation.

LAMBADA (Paperno et al. 2016, https://arxiv.org/abs/1606.06031) asks the model
to predict the FINAL WORD of a passage. Examples are selected so that a human
can guess the word given the whole paragraph but not from the last sentence
alone, which makes it a test of long-range context rather than local syntax.

Unlike HellaSwag there are no candidates to rank -- it is open-ended, so we
report two numbers:

    acc  - greedy decoding reproduces the final word exactly
    ppl  - exp(mean per-token loss) on the final word

We use EleutherAI/lambada_openai, the cleaned variant the GPT-2 and GPT-3
papers report on. The original HF `lambada` has detokenisation artifacts that
move scores by several points.

GPT-2 targets (paper, Table 3): 124M -> 45.99% acc / 35.13 ppl,
345M -> 55.48% / 15.60, 1.5B -> 63.24% / 8.63. Random accuracy is ~0%, so
unlike HellaSwag this is a signal you can actually watch climb.
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


def render_example(text):
    """One passage -> (tokens, n_target). The target is the final word.

    The target keeps its leading space: GPT-2 BPE encodes ' word' and 'word'
    differently, and only the former is how the word appears mid-sentence.
    About a quarter of final words span several BPE tokens, so n_target is
    not always 1.
    """
    context, last_word = text.rsplit(' ', 1)
    ctx_tokens = enc.encode(context)
    tgt_tokens = enc.encode(' ' + last_word)
    tokens = torch.tensor(ctx_tokens + tgt_tokens, dtype=torch.long).unsqueeze(0)
    return tokens, len(tgt_tokens)


@torch.inference_mode()
def score_example(model, tokens, n_target):
    """Return (hit, sum_loss, n_target) for one passage.

    hit is True only when greedy decoding reproduces EVERY target token --
    checking just the first would overstate accuracy on multi-token words.
    """
    tokens = tokens.to(device)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits, _ = model(tokens)

    # logits at position i predict token i+1, so targets and predictions both
    # shift by one (same alignment as hellaswag.score_example).
    shift_logits = logits[:, :-1, :].float()
    shift_tokens = tokens[:, 1:]

    # the final n_target positions of the shifted view are the target word
    tgt_logits = shift_logits[:, -n_target:, :]
    tgt_tokens = shift_tokens[:, -n_target:]

    hit = bool((tgt_logits.argmax(dim=-1) == tgt_tokens).all().item())
    sum_loss = F.cross_entropy(
        input=tgt_logits.reshape(-1, tgt_logits.size(-1)),
        target=tgt_tokens.reshape(-1),
        reduction='sum',
    ).item()
    return hit, sum_loss, n_target


def iter_examples(split='test'):
    from datasets import load_dataset
    return load_dataset('EleutherAI/lambada_openai', 'en', split=split)


def evaluate(model, limit=None, split='test', progress=False):
    """Score the model on LAMBADA. Returns (acc, ppl, n).

    Perplexity is exp(total loss / total target tokens) pooled over the whole
    set, not a mean of per-example perplexities -- the latter is dominated by
    a few catastrophic examples and is not what the papers report.

    Under DDP each rank takes every world_size-th example and the totals are
    summed before dividing, so every rank returns the same figures.
    """
    model.eval()
    n_total = n_correct = n_tokens = 0
    loss_total = 0.0

    examples = iter_examples(split)
    if limit is not None:
        examples = examples.select(range(min(limit, len(examples))))

    it = enumerate(examples)
    if progress and master_process:
        from tqdm import tqdm
        it = tqdm(it, total=len(examples), desc='LAMBADA', colour='#7BC621')

    for i, example in it:
        if i % ddp_world_size != ddp_rank:      # shard across ranks
            continue
        tokens, n_target = render_example(example['text'])
        if tokens.size(1) > model.config.block_size:
            tokens = tokens[:, -model.config.block_size:]
        hit, sum_loss, n_tgt = score_example(model, tokens, n_target)
        n_total += 1
        n_correct += int(hit)
        loss_total += sum_loss
        n_tokens += n_tgt

    if ddp:
        import torch.distributed as dist
        totals = torch.tensor([n_total, n_correct, n_tokens, loss_total],
                              dtype=torch.float64, device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        n_total, n_correct, n_tokens, loss_total = totals.tolist()

    import math
    return n_correct / n_total, math.exp(loss_total / n_tokens), int(n_total)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='uv run bardgpt-lambada',
        formatter_class=HelpFormatter,
        description=textwrap.dedent('''
        Evaluate a BardGPT checkpoint on LAMBADA.

        Predicts the final word of each passage and reports exact-match
        accuracy plus perplexity. GPT-2 124M scores 45.99% / 35.13 ppl.
        '''),
        epilog=textwrap.dedent('''
        Made with ♥︎  by Bardia Emamgholizadeh •︵•
        Linkedin: @bardiaegz | Telegram: @bardiaegz | Instagram: @bardia_egz
               ''')
    )
    parser.add_argument('-c', '--checkpoint', default='latest', metavar='PATH', help="checkpoint in checkpoint/ to evaluate; 'latest' picks the newest")
    parser.add_argument('-l', '--limit', default=None, metavar='N', type=int, help='only score the first N passages (default: all 5,153)')
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

    acc, ppl, n = evaluate(model, limit=args.limit, progress=True)

    print(f'\n{"=" * 25} LAMBADA {"=" * 26}')
    print(f'\npassages  : {n:,}')
    print(f'accuracy  : {colors.OKGREEN}{acc * 100:.2f}%{colors.ENDC}   (GPT-2 124M 45.99%)')
    print(f'perplexity: {colors.OKGREEN}{ppl:.2f}{colors.ENDC}   (GPT-2 124M 35.13)')
    print(f'\n{"=" * 60}')


if __name__ == '__main__':
    main()
