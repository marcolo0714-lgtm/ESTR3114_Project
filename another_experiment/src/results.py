"""Run naming, JSON I/O, and strict completed-run reuse."""
import csv
import json
import math
from pathlib import Path

METHODS = ('token_baseline', 'concept_token_only', 'joint', 'pcgrad')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def number(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Loss weights must be finite and nonnegative')
    return str(value)


def run_name(method, weight, beta, steps, seed):
    return f'{method}_lambda{number(weight)}_beta{number(beta)}_step{steps}_seed{seed}'


def completed_runs(root):
    for path in sorted(Path(root).glob('*/summary.json')):
        summary = read_json(path)
        if summary.get('complete'):
            yield path.parent, read_json(path.parent/'config.json'), summary


def index_runs(root):
    rows = []
    for path, config, summary in completed_runs(root):
        row = {k: config[k] for k in ('method', 'weight', 'beta', 'steps', 'seed')}
        row.update(path=str(path), elapsed_seconds=summary['elapsed_seconds'])
        for split in ('validation', 'test'):
            for metric in ('token_loss', 'perplexity', 'concept_loss', 'vq_loss'):
                if split in summary:
                    row[f'{split}_{metric}'] = summary[split][metric]
        rows.append(row)
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with open(Path(root)/'main_summary.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    return rows
