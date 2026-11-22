"""Atomic result files, reproducibility hashes, and completed-run indexing."""
import csv
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run_name(method, steps, seed):
    return f'{method}_step{steps}_seed{seed}'


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def settings_signature(config):
    values = {k:v for k,v in config.items() if k not in ('method', 'seed')}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:12]


def completed_runs(root):
    for path in sorted(Path(root).glob('*/summary.json')):
        summary = read_json(path)
        if summary.get('complete'):
            yield path.parent, read_json(path.parent/'config.json'), summary


def index_runs(root):
    rows = []
    for path, c, s in completed_runs(root):
        row = dict(method=c['method'], seed=c['seed'], steps=c['steps'],
                   settings_signature=settings_signature(c), run_directory=str(path),
                   elapsed_seconds=s['elapsed_seconds'], peak_gpu_memory_bytes=s['peak_gpu_memory_bytes'])
        for split in ('validation_final','test_final','test_best_validation'):
            if split in s:
                row.update({f'{split}_{k}':s[split][k] for k in
                            ('accuracy','macro_f1','loss','worst_task_accuracy')})
        rows.append(row)
    if rows:
        root = Path(root); root.mkdir(parents=True,exist_ok=True)
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with open(root/'main_summary.csv','w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    return rows
