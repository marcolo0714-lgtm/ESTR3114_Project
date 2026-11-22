"""Copy trusted text checkpoint ZIPs into site/models without external links.

Usage: python tools/import_models.py --archives <four ZIP paths> --backbone <JSON>
Only an explicit allowlist of ZIP members is written. No archive paths are used
as output destinations. Training output paths are removed from copied metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'token': 'Token-only', 'sentiment': 'Sentiment-only', 'joint': 'Joint', 'pcgrad': 'PCGrad'}
ALLOWED = ('final_model.pt', 'config.json', 'summary.json',
           'tokenizer/tokenizer_config.json', 'tokenizer/special_tokens_map.json',
           'tokenizer/tokenizer.json', 'tokenizer/vocab.json', 'tokenizer/merges.txt')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def clean_metadata(value):
    if isinstance(value, dict):
        return {k: clean_metadata(v) for k, v in value.items() if k not in ('output', 'model_snapshot')}
    if isinstance(value, list):
        return [clean_metadata(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, nargs='+', required=True)
    parser.add_argument('--backbone', type=Path, required=True)
    args = parser.parse_args()
    architecture = json.loads(args.backbone.read_text(encoding='utf-8'))
    if architecture.get('model_type') != 'gpt2':
        raise ValueError('Expected a GPT-2 architecture configuration')
    models_dir = ROOT/'models'
    text_dir = models_dir/'text'
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir/'backbone_config.json').write_text(json.dumps(architecture, indent=2)+'\n', encoding='utf-8')
    entries = {}
    for archive in args.archives:
        with ZipFile(archive) as z:
            configs = [n for n in z.namelist() if n.endswith('/config.json') and '/tokenizer/' not in n]
            if len(configs) != 1:
                raise ValueError(f'Expected exactly one run in {archive.name}')
            cfg = json.loads(z.read(configs[0]))
            method = cfg['method']
            if method not in NAMES or cfg['steps'] != 6000 or cfg['seed'] != 44 or cfg['max_length'] != 96:
                raise ValueError('Expected one of the seed-44, 6000-step, 96-token demo models')
            if method in ('joint', 'pcgrad') and cfg['weight'] != 3.0:
                raise ValueError('Expected sentiment weight 3 for multitask models')
            if method in entries:
                raise ValueError(f'Duplicate method {method}')
            prefix = configs[0][:-len('config.json')]
            destination = text_dir/method
            destination.mkdir(parents=True, exist_ok=True)
            for name in ALLOWED:
                source = prefix + name
                if source not in z.namelist():
                    raise ValueError(f'Missing required member {name}')
                if z.namelist().count(source) != 1:
                    raise ValueError(f'Duplicate archive member {name}')
                target = destination/name
                target.parent.mkdir(parents=True, exist_ok=True)
                if name in ('config.json', 'summary.json'):
                    target.write_text(json.dumps(clean_metadata(json.loads(z.read(source))), indent=2)+'\n', encoding='utf-8')
                else:
                    staging = target.with_name(target.name+'.importing')
                    with z.open(source) as src, staging.open('wb') as dst:
                        shutil.copyfileobj(src, dst, length=4*1024*1024)
                    staging.replace(target)
            entries[method] = dict(id=method, name=NAMES[method], seed=44, steps=6000,
                                   weight=cfg['weight'], max_length=96,
                                   sentiment_trained=method != 'token',
                                   checkpoint=f'text/{method}/final_model.pt',
                                   tokenizer=f'text/{method}/tokenizer',
                                   architecture='text/backbone_config.json',
                                   checkpoint_sha256=sha256(destination/'final_model.pt'))
            print(f'Imported and hashed {NAMES[method]}', flush=True)
    if set(entries) != set(NAMES):
        raise ValueError('Supply all four method archives together')
    registry = dict(models=[entries[k] for k in NAMES])
    (models_dir/'registry.json').write_text(json.dumps(registry, indent=2)+'\n', encoding='utf-8')
    print('Four local text checkpoints ready.', flush=True)


if __name__ == '__main__':
    main()
