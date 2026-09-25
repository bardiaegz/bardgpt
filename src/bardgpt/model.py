import torch
import torch.nn as nn
import torch.nn.functional as F

# RMSNorm: Root Mean Square Layer Normalization -> https://arxiv.org/pdf/1910.07467
class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.eps = config.eps

    def forward(self, x: torch.Tensor):
        rms = torch.rsqrt(self.eps + (x.pow(2).mean(dim=-1, keepdim=True)))
        y = self.weight * rms * x
        return y

# SwiGLU: GLU Variants Improve Transformer -> https://arxiv.org/pdf/2002.05202
class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 2/3 factor from LLaMA paper §2.2
        # keeps params equal to a 2-matrix FFN, since SwiGLU needs 3: 3 * 2/3 * 4d == 2 * 4d
        # (LLaMA): Open and Efficient Foundation Language Models -> https://arxiv.org/abs/2302.13971
        hidden_dim = int(2 / 3 * config.expansion_factor * config.n_embd)
        self.w1 = nn.Linear(in_features=config.n_embd, out_features=hidden_dim)
        self.w2 = nn.Linear(in_features=config.n_embd, out_features=hidden_dim)
        self.w3 = nn.Linear(in_features=hidden_dim, out_features=config.n_embd)
        self.w3.BARDGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# Transformer: Attention Is All You Need -> https://arxiv.org/abs/1706.03762 
# GPT-1: Improving Language Understanding by Generative Pre-Training -> https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
# GPT-2: Language Models are Unsupervised Multitask Learners -> https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
# GPT-3: Language Models are Few-Shot Learners -> https://arxiv.org/pdf/2005.14165
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

        norm, activation = config.norm, config.activation
        assert norm in {'layer_norm', 'rmsnorm'}
        assert activation in {'gelu', 'swiglu'}

        if norm == 'layer_norm':
            self.ln_1 = nn.LayerNorm(normalized_shape=config.n_embd)
            self.ln_2 = nn.LayerNorm(normalized_shape=config.n_embd)

        elif norm == 'rmsnorm':
            self.ln_1 = RMSNorm(config=config)
            self.ln_2 = RMSNorm(config=config)

        self.attn = CausalSelfAttention(config=config)

        if activation == 'gelu':
            self.mlp = MLP(config=config)
        elif activation == 'swiglu':
            self.mlp = SwiGLU(config=config)
            
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class BardGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        assert config.norm in {'layer_norm', 'rmsnorm'}
        assert config.activation in {'gelu', 'swiglu'}
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.n_embd),
            wpe = nn.Embedding(num_embeddings=config.block_size, embedding_dim=config.n_embd),
            h = nn.ModuleList(Block(config=config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(normalized_shape=config.n_embd) if config.norm == 'layer_norm' else RMSNorm(config),
        ))

        self.lm_head = nn.Linear(in_features=config.n_embd, out_features=config.vocab_size, bias=False)

        # weight tying: eq. (2) in GPT-1 paper uses the same W_e for the input
        # embedding and the output projection -> P(u) = softmax(h_n W_e^T).
        # Using the Output Embedding to Improve Language Models -> https://arxiv.org/abs/1608.05859
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    # std=0.02 from GPT-1 §4.1 (Model specifications);
    # residual scaling 1/sqrt(N) from GPT-2 §2.3 (Model), N = 2 * n_layer
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
        # Betas and Epsilon from Appendix B on GPT-3 paper.
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused)