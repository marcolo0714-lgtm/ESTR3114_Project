"""Dataset loading and right-padded batches (GPT-2 uses absolute positions)."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class TweetDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int, limit: int = 0):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if limit:
            rows = rows[:limit]  # Consistent deterministic pilot subset.
        if not rows:
            raise ValueError(f"Empty dataset: {path}")
        self.examples = []
        for row in rows:
            # Explicit EOS supplies a predictable final text token. At least two
            # tokens are needed to define a next-token target for short tweets.
            ids = tokenizer.encode(row["text"], add_special_tokens=False)
            ids = ids[: max_length - 1]
            ids = (ids or [tokenizer.eos_token_id]) + [tokenizer.eos_token_id]
            self.examples.append({"input_ids": ids, "label": int(row["label"])})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def make_collator(pad_token_id: int):
    def collate(examples):
        length = max(len(row["input_ids"]) for row in examples)
        ids = torch.full((len(examples), length), pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(examples):
            tokens = row["input_ids"]
            ids[i, : len(tokens)] = torch.tensor(tokens)
            mask[i, : len(tokens)] = 1
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": torch.tensor([row["label"] for row in examples]),
        }
    return collate
