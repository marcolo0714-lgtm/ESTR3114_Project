"""Bundle trusted experiment-2 checkpoints and compact report evidence inside site.

Usage: python tools/import_image_models.py --results ../new_experiment/results --analysis ../new_experiment/analysis
This explicit preparation step is the only step that reads the training project.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.image_model import ModelConfig, MultitaskCNN


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    registry = dict(models=[])
    pairs = []
    for method, name in [('joint', 'Joint'), ('pcgrad', 'PCGrad')]:
        directory = args.results/f'{method}_step10000_seed42'
        config = read(directory/'config.json')
        summary = read(directory/'summary.json')
        history = [json.loads(x) for x in (directory/'history.jsonl').read_text().splitlines()]
        validation = [json.loads(x) for x in (directory/'validation.jsonl').read_text().splitlines()]
        assert summary['complete'] and config['method'] == method
        assert config['steps'] == 10000 and config['seed'] == 42
        assert [x['step'] for x in history] == list(range(1,10001))
        pairs.append([x['batch_sha256'] for x in history])
        models = ROOT/'models/images'
        entry = dict(id=method, name=name, seed=42, training_steps=10000, checkpoints={})
        for kind in ('best_validation', 'final'):
            source = directory/'checkpoints'/f'{kind}.pt'
            saved = torch.load(source, map_location='cpu', weights_only=True)
            assert saved['config'] == config
            expected = summary['best_validation_step'] if kind == 'best_validation' else 10000
            assert saved['step'] == expected
            model = MultitaskCNN(ModelConfig(**saved['model_config']))
            model.load_state_dict(saved['model'], strict=True)
            tasks = saved['mapping']['tasks']
            normalization = config['assets']['normalization']
            if 'tasks' in registry:
                assert registry['tasks'] == tasks and registry['normalization'] == normalization
            registry.update(tasks=tasks, normalization=normalization)
            bundle = dict(format_version=1, model=saved['model'], model_config=saved['model_config'],
                          mapping=saved['mapping'], normalization=normalization, method=method,
                          seed=42, step=saved['step'], training_steps=10000, checkpoint_kind=kind,
                          source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            target = models/method/f'{kind}.pt'
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.importing')
            torch.save(bundle, temporary)
            temporary.replace(target)
            entry['checkpoints'][kind] = dict(path=f'{method}/{kind}.pt', step=saved['step'],
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(), source_sha256=bundle['source_sha256'])
        registry['models'].append(entry)
        diagnostics = [x for x in history if 'pairwise_cosines' in x]
        mask = ~np.eye(20, dtype=bool)
        diag = {k:float(np.mean([x[k] for x in diagnostics]))
                for k in ('conflict_fraction','mean_absolute_cosine','projection_relative_norm')}
        diag.update(samples=len(diagnostics),
                    mean_cosines=np.mean([x['pairwise_cosines'] for x in diagnostics],axis=0).tolist(),
                    post_conflict_fraction=float(np.mean([np.mean(np.array(x['post_pairwise_cosines'])[mask]<0)
                                                          for x in diagnostics])))
        by_step = {x['step']:x for x in history}
        curves = [dict(step=x['step'], accuracy=x['accuracy'], macro_f1=x['macro_f1'], loss=x['loss'],
                       elapsed_seconds=by_step[x['step']]['elapsed_seconds']) for x in validation]
        write(ROOT/f'experiment_data/images/{method}.json', dict(config=config, summary=summary,
              curves=curves, diagnostics=diag,
              source_sha256={n:hashlib.sha256((directory/n).read_bytes()).hexdigest()
                             for n in ('config.json','summary.json','history.jsonl','validation.jsonl')}))
    assert pairs[0] == pairs[1], 'Sampling and augmentation must match'
    write(ROOT/'models/images/registry.json', registry)
    figures = ROOT/'static/figures/cifar'
    figures.mkdir(parents=True, exist_ok=True)
    for name in ('comparison_10000_seed42.png','task_differences_10000_seed42.png',
                 'summary_runtime.png','gradient_cosines_joint_6bd2e268b095.png',
                 'gradient_cosines_pcgrad_6bd2e268b095.png'):
        shutil.copy2(args.analysis/name, figures/name)
    print('Imported four image checkpoint states and matched seed-42 report evidence.')


if __name__ == '__main__': main()
