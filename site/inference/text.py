"""Lazy, offline checkpoint loading and serialized inference for a small demo."""
import json
from pathlib import Path
from threading import Lock
from time import perf_counter

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2Model

from .model import JointGPT2

LABELS = ('negative', 'neutral', 'positive')


def prepare_input(tokenizer, text, max_length=96):
    ids = tokenizer.encode(text, add_special_tokens=False)
    original_count = len(ids)
    ids = ids[:max_length - 1]
    # Matches the original data pipeline even for unusual inputs with no tokens.
    ids = (ids or [tokenizer.eos_token_id]) + [tokenizer.eos_token_id]
    return ids, original_count, original_count > max_length - 1


class ModelStore:
    def __init__(self, models_dir, device='cpu'):
        self.models_dir = Path(models_dir).resolve()
        self.registry = json.loads((self.models_dir / 'registry.json').read_text(encoding='utf-8'))
        self.entries = {entry['id']: entry for entry in self.registry['models']}
        if device not in ('cpu', 'cuda', 'auto'):
            raise ValueError('Device must be cpu, cuda, or auto')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; use --device cpu')
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available()
                                   else 'cpu' if device == 'auto' else device)
        self.loaded = {}
        self.lock = Lock()

    def local_path(self, relative):
        path = (self.models_dir / relative).resolve()
        if not path.is_relative_to(self.models_dir):
            raise ValueError('Model path must stay inside site/models')
        return path

    def available(self, entry):
        return all(self.local_path(entry[k]).exists() for k in ('checkpoint', 'tokenizer', 'architecture'))

    def status(self):
        return [dict(id=e['id'], name=e['name'], seed=e['seed'], steps=e['steps'],
                     weight=e['weight'], sentiment_trained=e['sentiment_trained'],
                     available=self.available(e), loaded=e['id'] in self.loaded)
                for e in self.entries.values()]

    def _load(self, key):
        if key in self.loaded:
            return self.loaded[key]
        entry = self.entries[key]
        if not self.available(entry):
            raise FileNotFoundError('This model has not been installed')
        architecture = GPT2Config.from_json_file(str(self.local_path(entry['architecture'])))
        model = JointGPT2(GPT2Model(architecture))
        state = torch.load(self.local_path(entry['checkpoint']), map_location='cpu',
                           weights_only=True, mmap=True)
        model.load_state_dict(state, strict=True)
        del state
        model.to(self.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.local_path(entry['tokenizer']), local_files_only=True)
        self.loaded[key] = (model, tokenizer)
        return model, tokenizer

    def predict(self, text, methods):
        """Keep requests sequential: avoid concurrent CPU work and duplicate loads."""
        with self.lock:
            total_start = perf_counter()
            results = []
            for key in methods:
                entry = self.entries[key]
                was_loaded = key in self.loaded
                start = perf_counter()
                model, tokenizer = self._load(key)
                loading_ms = (perf_counter() - start) * 1000 if not was_loaded else 0
                ids, original_count, truncated = prepare_input(tokenizer, text, entry['max_length'])
                input_ids = torch.tensor([ids], device=self.device)
                attention_mask = torch.ones_like(input_ids)
                start = perf_counter()
                with torch.inference_mode():
                    token_logits, sentiment_logits = model.predict_positions(input_ids, attention_mask)
                    probabilities, indices = token_logits[0].softmax(-1).topk(5)
                    token_probs = probabilities.cpu().tolist()
                    token_ids = indices.cpu().tolist()
                    sentiment = sentiment_logits[0].softmax(-1).cpu().tolist()
                inference_ms = (perf_counter() - start) * 1000
                tokens = [dict(id=i, text=tokenizer.decode([i], clean_up_tokenization_spaces=False),
                               probability=p, is_eos=i == tokenizer.eos_token_id)
                          for i, p in zip(token_ids, token_probs)]
                trained = entry['sentiment_trained']
                results.append(dict(
                    id=key, name=entry['name'], tokens=tokens,
                    sentiment=dict(label=LABELS[max(range(3), key=sentiment.__getitem__)],
                                   probabilities=dict(zip(LABELS, sentiment))) if trained else None,
                    original_tokens=original_count, used_text_tokens=len(ids)-1, truncated=truncated,
                    processed_text=tokenizer.decode(ids[:-1], clean_up_tokenization_spaces=False),
                    inference_ms=round(inference_ms, 1), loading_ms=round(loading_ms, 1),
                    first_load=not was_loaded))
            return dict(results=results, total_ms=round((perf_counter()-total_start)*1000, 1),
                        device=self.device.type, seed=44, steps=6000)
