"""Document-separated splits, stable hashes, and fixed-length local token blocks."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_documents(path):
    """Local JSONL: one {text: ...} document per line; TXT: blank-line separated."""
    path = Path(path)
    if path.suffix == '.jsonl':
        return [json.loads(line)['text'] for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return [s.strip() for s in path.read_text(encoding='utf-8').split('\n\n') if s.strip()]


def split_documents(documents, seed):
    # Deduplicate before assigning splits, including across published split files.
    unique = list(dict.fromkeys(s.strip() for s in documents if s.strip()))
    if len(unique) < 20:
        raise ValueError('At least 20 distinct documents are required')
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n = len(unique)
    a, b = int(n*.8), int(n*.9)
    return dict(train=unique[:a], validation=unique[a:b], test=unique[b:])


def write_blocks(documents, tokenizer, path, length):
    """Pack within a split, append EOS, retain the tail with a validity mask."""
    tokens = []
    for doc in documents:
        tokens.extend(tokenizer.encode(doc, add_special_tokens=False))
        tokens.append(tokenizer.eos_token_id)
    if len(tokens) < length:
        raise ValueError(f'{path}: split is too small for one sequence')
    count = (len(tokens)+length-1)//length
    ids = np.full((count, length), tokenizer.eos_token_id, dtype=np.int32)
    mask = np.zeros_like(ids, dtype=np.uint8)
    flat = ids.reshape(-1)
    flat[:len(tokens)] = tokens
    mask.reshape(-1)[:len(tokens)] = 1
    np.savez(path, input_ids=ids, attention_mask=mask)
    return dict(documents=len(documents), tokens=len(tokens), blocks=count, sha256=sha256(path))


class Blocks(Dataset):
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.ids = torch.from_numpy(data['input_ids'].astype(np.int64))
            self.mask = torch.from_numpy(data['attention_mask'].astype(np.bool_))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return {'input_ids': self.ids[index], 'attention_mask': self.mask[index]}
