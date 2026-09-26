import torch
import os
from torch.distributed import init_process_group

ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    assert torch.cuda.is_available(), 'You need CUDA for DDP.'
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    master_process = ddp_rank == 0
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device=device)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    print(f'using device: {device}')


device_type: str = device.split(':')[0]

