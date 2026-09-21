import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(in_features=config.n_embd, out_features=3 * config.n_embd)
        self.c_proj = nn.Linear(in_features=config.n_embd, out_features=config.n_embd)
        self.c_proj.BARDGPT_SCALE_INIT = 1
        self.n_embd = config.n_embd
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(in_features=config.n_embd, out_features=config.expansion_factor * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(in_features=config.expansion_factor * config.n_embd, out_features=config.n_embd)
        self.c_proj.BARDGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(normalized_shape=config.n_embd)
        self.attn = CausalSelfAttention(config=config)
        self.ln_2 = nn.LayerNorm(normalized_shape=config.n_embd)
        self.mlp = MLP(config=config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class BardGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.n_embd),
            wpe = nn.Embedding(num_embeddings=config.block_size, embedding_dim=config.n_embd),
            h = nn.ModuleList(Block(config=config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(normalized_shape=config.n_embd),
        ))

        self.lm_head = nn.Linear(in_features=config.n_embd, out_features=config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'BARDGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(tensor=module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(tensor=module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(tensor=module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f'cannot forward a sequence of length T ({T}), it should be less than block size ({self.config.block_size})'
        pos = torch.arange(start=0, end=T, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(input=logits.view(-1, logits.size(-1)), target=targets.view(-1))
        return logits, loss

    def configure_optimizer(self, weight_decay, learning_rate, device_type):
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        
        print(f'num decay parameters: {sum(p.numel() for p in decay):,}')
        print(f'num no-decay parameters: {sum(p.numel() for p in nodecay):,}')
                        
        groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': nodecay, 'weight_decay': 0.0},
        ]
        fused = device_type == 'cuda'
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused)