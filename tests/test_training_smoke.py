"""Exercise the real training loop with an in-memory miniature backbone.

This catches integration mistakes without downloading the large GPT-2 model.
"""

import argparse
import json
import sys
from types import SimpleNamespace

import torch
from torch import nn

import train


class TinyTokenizer:
    eos_token_id = 15
    eos_token = "<eos>"
    pad_token = None
    pad_token_id = 15

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def encode(self, text, add_special_tokens=False):
        return [(ord(char) % 14) + 1 for char in text]


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_embd=4, vocab_size=16)
        self.wte = nn.Embedding(16, 4)
        self.projection = nn.Linear(4, 4)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def forward(self, input_ids, attention_mask, **kwargs):
        return SimpleNamespace(last_hidden_state=self.projection(self.wte(input_ids)))


def test_pcgrad_training_loop_writes_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=TinyTokenizer, GPT2Model=TinyBackbone
    ))
    model_dir = tmp_path / "resources" / "model"
    data_dir = tmp_path / "resources" / "data"
    model_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    rows = [
        {"text": "good", "label": 2}, {"text": "bad", "label": 0},
        {"text": "okay", "label": 1}, {"text": "nice", "label": 2},
    ]
    for split in ("train", "validation"):
        (data_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    output = tmp_path / "run"
    args = argparse.Namespace(
        method="pcgrad", weight=0.3, seed=42, steps=2,
        batch_size=2, eval_batch_size=2, max_length=8,
        learning_rate=1e-3, weight_decay=0.01, warmup_ratio=0.1,
        eval_every=1, diagnostics_every=1, train_limit=0,
        validation_limit=0, test=False, output=output,
        device="cpu", save_model=False,
    )
    summary = train.run(args)
    history = [json.loads(line) for line in (output / "history.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len(history) == 2
    assert summary["validation"]["examples"] == 4
    assert summary["validation"]["tokens"] > 0
    assert "layer_cosines" in history[0]["gradients"]
    assert 0 <= summary["validation"]["macro_f1"] <= 1
