"""Pretrained GPT-2 backbone with two genuinely separate task heads."""

import torch
from torch import nn
from torch.nn import functional as F


class JointGPT2(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.n_embd
        vocab = backbone.config.vocab_size
        self.token_head = nn.Linear(hidden, vocab, bias=False)
        self.sentiment_head = nn.Linear(hidden, 3)
        # GPT-2's pretrained language head is tied to its input embeddings.
        # Copy its initial values but UNTIE it so head parameters receive only
        # their own task gradient, while embeddings remain shared parameters.
        with torch.no_grad():
            self.token_head.weight.copy_(backbone.wte.weight)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, return_dict=True,
        ).last_hidden_state
        # Gather EOS, the final real token of each right-padded example.
        last = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        return self.token_head(hidden), self.sentiment_head(pooled)


def task_losses(token_logits, sentiment_logits, batch):
    """Token loss averages over real targets only, never padded positions."""
    targets = batch["input_ids"][:, 1:]
    valid = batch["attention_mask"][:, 1:].bool()
    per_token = F.cross_entropy(
        token_logits[:, :-1].reshape(-1, token_logits.size(-1)),
        targets.reshape(-1), reduction="none",
    ).reshape_as(targets)
    token_sum = (per_token * valid).sum()
    token_count = valid.sum()
    token_loss = token_sum / token_count.clamp_min(1)
    sentiment_sum = F.cross_entropy(sentiment_logits, batch["labels"], reduction="sum")
    sentiment_loss = sentiment_sum / batch["labels"].numel()
    return token_loss, sentiment_loss, token_sum.detach(), token_count.detach(), sentiment_sum.detach()
