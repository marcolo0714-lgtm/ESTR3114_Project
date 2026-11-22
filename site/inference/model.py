"""Exact parameter structure of the experiment's GPT-2 with private task heads."""
import torch
from torch import nn


class JointGPT2(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.token_head = nn.Linear(backbone.config.n_embd, backbone.config.vocab_size, bias=False)
        self.sentiment_head = nn.Linear(backbone.config.n_embd, 3)
        with torch.no_grad():
            self.token_head.weight.copy_(backbone.wte.weight)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False, return_dict=True).last_hidden_state
        last = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        return self.token_head(hidden), self.sentiment_head(pooled)

    def predict_positions(self, input_ids, attention_mask):
        """One forward pass: next-token logits BEFORE EOS; sentiment AT EOS.

        There is one unpadded example. Computing the vocabulary projection only
        at the final text position avoids allocating logits for the entire text.
        """
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False, return_dict=True).last_hidden_state
        return self.token_head(hidden[:, -2]), self.sentiment_head(hidden[:, -1])
