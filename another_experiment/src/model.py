"""Small ConceptLM adaptation, implemented in PyTorch without parent imports.

GPT-2 weights initialize token layers; concept layers are freshly initialized.
Codebooks receive only VQ gradients. Targets and codebook lookup inputs detach.
This explicit gradient policy is an adaptation, not an exact paper reproduction.
"""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .quantizer import ProductQuantizer
from .losses import masked_mean


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    hidden_size: int = 768
    token_layers: int = 12
    heads: int = 12
    max_positions: int = 1024
    encoder_layers: int = 1
    concept_layers: int = 2
    chunk_size: int = 4
    codebook_size: int = 64
    segments: int = 12
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5

    def __post_init__(self):
        for name in ('vocab_size', 'hidden_size', 'token_layers', 'heads',
                     'max_positions', 'concept_layers', 'chunk_size',
                     'codebook_size', 'segments'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.hidden_size % self.heads or self.hidden_size % self.segments:
            raise ValueError('hidden_size must divide evenly into heads and segments')
        if not 1 <= self.encoder_layers < self.token_layers:
            raise ValueError('encoder_layers must be between 1 and token_layers - 1')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.dropout = c.heads, c.dropout
        self.c_attn = nn.Linear(c.hidden_size, 3 * c.hidden_size)
        self.c_proj = nn.Linear(c.hidden_size, c.hidden_size)
        self.resid_dropout = nn.Dropout(c.dropout)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = [z.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
                   for z in self.c_attn(x).chunk(3, dim=-1)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                    dropout_p=self.dropout if self.training else 0.0)
        return self.resid_dropout(self.c_proj(y.transpose(1, 2).reshape(b, t, d)))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_fc = nn.Linear(c.hidden_size, 4 * c.hidden_size)
        self.c_proj = nn.Linear(4 * c.hidden_size, c.hidden_size)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x):
        # Hugging Face GPT-2's gelu_new is the tanh approximation.
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate='tanh')))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln_1 = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_epsilon)
        self.attn = Attention(c)
        self.ln_2 = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_epsilon)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class ConceptModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        self.wte = nn.Embedding(c.vocab_size, c.hidden_size)
        self.wpe = nn.Embedding(c.max_positions, c.hidden_size)
        self.drop = nn.Dropout(c.dropout)
        self.h = nn.ModuleList([Block(c) for _ in range(c.token_layers)])
        self.ln_f = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_epsilon)
        self.concept_blocks = nn.ModuleList([Block(c) for _ in range(c.concept_layers)])
        self.concept_norm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_epsilon)
        self.concept_head = nn.Linear(c.hidden_size, c.segments * c.codebook_size)
        self.quantizer = ProductQuantizer(c.hidden_size, c.segments, c.codebook_size)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def load_gpt2(self, hf_model):
        """Copy HF Conv1D weights into equivalent nn.Linear token layers."""
        source = hf_model.transformer.state_dict()
        target = self.state_dict()
        required = {name for name in target if name.startswith(('wte.', 'wpe.', 'h.', 'ln_f.'))}
        missing = required - source.keys()
        if missing:
            raise ValueError(f'GPT-2 checkpoint is missing token tensors: {sorted(missing)}')
        for name, value in source.items():
            if name not in target:
                continue
            if name.endswith(('c_attn.weight', 'c_fc.weight', 'c_proj.weight')):
                value = value.t()
            if value.shape != target[name].shape:
                raise ValueError(f'Incompatible GPT-2 tensor: {name}')
            target[name] = value
        self.load_state_dict(target)

    def forward(self, input_ids, attention_mask=None, use_concepts=True):
        c = self.config
        b, t = input_ids.shape
        if t > c.max_positions or t % c.chunk_size or t < 2 * c.chunk_size:
            raise ValueError('length must fit max_positions and contain >=2 complete chunks')
        mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        x = self.drop(self.wte(input_ids) + self.wpe(torch.arange(t, device=input_ids.device)))
        for block in self.h[:c.encoder_layers]:
            x = block(x)
        zero = x.sum() * 0
        ncp, vq, stats = zero, zero, {}
        if use_concepts:
            m, k = t // c.chunk_size, c.chunk_size
            pooled = x.reshape(b, m, k, -1).mean(2)
            valid = mask.reshape(b, m, k).all(2)
            vq, stats = self.quantizer(pooled, valid)
            z = pooled
            for block in self.concept_blocks:
                z = block(z)
            probabilities = self.concept_head(self.concept_norm(z)).reshape(
                b, m, c.segments, c.codebook_size).softmax(-1)
            predicted = torch.einsum('bmsn,snd->bmsd', probabilities,
                                     self.quantizer.embeddings().detach()).flatten(-2)
            errors = (predicted[:, :-1] - pooled[:, 1:].detach()).square().mean(-1)
            ncp = masked_mean(errors, valid[:, :-1] & valid[:, 1:])
            # Position k-1 predicts token k: only the preceding complete chunk
            # is available. No ground-truth future representation enters decoding.
            positions = torch.arange(t, device=x.device)
            indices = ((positions + 1) // k - 1).clamp_min(0)
            guidance = predicted[:, indices]
            x = x + guidance * (positions >= k - 1)[None, :, None]
        for block in self.h[c.encoder_layers:]:
            x = block(x)
        logits = F.linear(self.ln_f(x), self.wte.weight)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, c.vocab_size),
                             input_ids[:, 1:].reshape(-1), reduction='none').reshape(b, t-1)
        token_mask = mask[:, :-1] & mask[:, 1:]
        return dict(logits=logits, token_loss=masked_mean(ce, token_mask),
                    token_count=token_mask.sum(), concept_loss=ncp, vq_loss=vq,
                    concept_count=(valid[:, :-1] & valid[:, 1:]).sum() if use_concepts else mask.sum()*0,
                    **stats)

    def save_config(self):
        return asdict(self.config)
