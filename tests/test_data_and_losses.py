"""Padding must not change the losses of shorter tweets."""

import torch

from src.data import make_collator
from src.model import task_losses


def test_right_padding_and_shifted_token_targets():
    batch = make_collator(99)([
        {"input_ids": [1, 2, 3], "label": 0},
        {"input_ids": [4, 5], "label": 2},
    ])
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 99]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    logits = torch.zeros((2, 3, 100))
    logits[0, 0, 2] = 4  # predicts second token
    logits[0, 1, 3] = 4  # predicts third token
    logits[1, 0, 5] = 4  # predicts second token
    sentiment = torch.zeros((2, 3))
    loss, _, total, count, _ = task_losses(logits, sentiment, batch)
    assert count.item() == 3
    assert torch.allclose(loss, total / 3)
    logits[1, 1, 99] = 1000  # prediction at a padded target is ignored
    assert torch.allclose(task_losses(logits, sentiment, batch)[0], loss)
