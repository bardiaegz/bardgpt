from dataclasses import dataclass
import torch
import os

proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
@dataclass
class BardGPTConfig:
    vocab_size: int = 50_257
    block_size: int = 1_024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    expansion_factor: int = 4
    eps: float = 1e-5

class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    OKGREEN = '\033[38;2;123;198;33m'

device = torch.accelerator.current_accelerator()
device_type: str = device.type
B = 4
T = 32
max_steps = 500
warmup_steps = 20
max_lr = 6e-4
min_lr = 0.1 * max_lr
weight_decay = 0.1
temperature = 1
k = 50
p = 0.9