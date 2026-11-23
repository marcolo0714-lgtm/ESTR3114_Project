"""Bounded in-memory image decoding and offline conditional CNN inference."""
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch

from .image_model import ModelConfig, MultitaskCNN

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def decode_image(data):
    """Match the CLI's RGB/bilinear preprocessing, honoring camera orientation."""
    if not data:
        raise ValueError('Choose an image first.')
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError('Choose an image no larger than 8 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in ('JPEG', 'PNG', 'WEBP'):
                    raise ValueError('Use a PNG, JPEG or WebP image.')
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ValueError('Choose an image with at most 20 million pixels.')
                source.load()
                image = ImageOps.exif_transpose(source).convert('RGB')
                size = image.size
                return image.resize((32, 32), Image.Resampling.BILINEAR), size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError('This image could not be read. Use a valid PNG, JPEG or WebP image.') from exc


class ImageModelStore:
    def __init__(self, models_dir, device='cpu'):
        self.root = Path(models_dir).resolve()
        self.registry = json.loads((self.root/'registry.json').read_text(encoding='utf-8'))
        self.entries = {x['id']: x for x in self.registry['models']}
        self.tasks = self.registry['tasks']
        if device not in ('cpu', 'cuda', 'auto'):
            raise ValueError('Unknown device')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable')
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available()
                                   else 'cpu' if device == 'auto' else device)
        self.loaded = {}
        self.lock = Lock()

    def local_path(self, relative):
        path = (self.root/relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Model path must stay inside the image models folder')
        return path

    def status(self):
        return dict(tasks=self.tasks, models=[dict(
            id=e['id'], name=e['name'], seed=e['seed'], training_steps=e['training_steps'],
            checkpoints={kind: dict(step=c['step'], available=self.local_path(c['path']).is_file(),
                                   loaded=(e['id'], kind) in self.loaded)
                         for kind, c in e['checkpoints'].items()}) for e in self.entries.values()])

    def _load(self, method, kind):
        key = method, kind
        if key in self.loaded:
            return self.loaded[key]
        entry = self.entries[method]['checkpoints'][kind]
        path = self.local_path(entry['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError('Image checkpoint integrity check failed')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if (saved['method'], saved['checkpoint_kind'], saved['step']) != (method, kind, entry['step']):
            raise RuntimeError('Image checkpoint metadata mismatch')
        if saved['mapping']['tasks'] != self.tasks:
            raise RuntimeError('Image checkpoint labels mismatch')
        if saved['normalization'] != self.registry['normalization']:
            raise RuntimeError('Image checkpoint preprocessing mismatch')
        model = MultitaskCNN(ModelConfig(**saved['model_config']))
        model.load_state_dict(saved['model'], strict=True)
        model.to(self.device).eval()
        self.loaded[key] = model
        return model

    def predict(self, data, task, methods, kind='best_validation'):
        if task not in range(len(self.tasks)):
            raise ValueError('Choose a valid superclass.')
        methods = list(dict.fromkeys(methods))
        if not methods or any(m not in self.entries for m in methods):
            raise ValueError('Select Joint, PCGrad, or both.')
        if kind not in ('best_validation', 'final'):
            raise ValueError('Choose a valid checkpoint.')
        total_start = perf_counter()
        image, original_size = decode_image(data)
        x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().unsqueeze(0)/255
        norm = self.registry['normalization']
        x = (x-torch.tensor(norm['mean']).reshape(1,3,1,1))/torch.tensor(norm['std']).reshape(1,3,1,1)
        preview = BytesIO()
        image.save(preview, format='PNG')
        results = []
        with self.lock:
            x = x.to(self.device)
            for method in methods:
                entry = self.entries[method]
                was_loaded = (method, kind) in self.loaded
                start = perf_counter()
                model = self._load(method, kind)
                if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
                loading_ms = (perf_counter()-start)*1000
                start = perf_counter()
                with torch.inference_mode():
                    values = model(x, task).softmax(-1)[0].cpu().tolist()
                inference_ms = (perf_counter()-start)*1000
                labels = self.tasks[task]['class_names']
                probabilities = sorted([dict(label=label, probability=value)
                    for label, value in zip(labels, values)], key=lambda v: -v['probability'])
                results.append(dict(id=method, name=entry['name'], seed=entry['seed'],
                    checkpoint=kind, step=entry['checkpoints'][kind]['step'],
                    predicted_class=probabilities[0]['label'], probabilities=probabilities,
                    inference_ms=round(inference_ms, 2), loading_ms=round(loading_ms, 2),
                    first_load=not was_loaded))
        return dict(results=results, task=self.tasks[task], checkpoint=kind,
            image=dict(width=original_size[0], height=original_size[1],
                       preview='data:image/png;base64,'+base64.b64encode(preview.getvalue()).decode('ascii')),
            total_ms=round((perf_counter()-total_start)*1000, 2), device=self.device.type)
