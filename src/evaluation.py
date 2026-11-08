"""Weighted token loss and tweet-level sentiment metrics."""

import math

import torch
from sklearn.metrics import accuracy_score, f1_score

from .model import task_losses


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    token_sum = token_count = sentiment_sum = examples = 0
    labels, predictions = [], []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        token_logits, sentiment_logits = model(batch["input_ids"], batch["attention_mask"])
        _, _, tok_sum, tok_count, sent_sum = task_losses(token_logits, sentiment_logits, batch)
        token_sum += tok_sum.item()
        token_count += tok_count.item()
        sentiment_sum += sent_sum.item()
        examples += batch["labels"].numel()
        labels.extend(batch["labels"].cpu().tolist())
        predictions.extend(sentiment_logits.argmax(dim=-1).cpu().tolist())
    model.train()
    token_loss = token_sum / max(token_count, 1)
    return {
        "token_loss": token_loss,
        "perplexity": math.exp(min(token_loss, 80)),
        "sentiment_loss": sentiment_sum / max(examples, 1),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0),
        "examples": examples,
        "tokens": token_count,
    }
