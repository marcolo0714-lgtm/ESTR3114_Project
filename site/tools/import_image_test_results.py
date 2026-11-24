"""Bundle held-out test metrics without dataset files, weights or predictions."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=ROOT.parent/'new_experiment')
    args = parser.parse_args()
    timing = json.loads((args.experiment/'analysis/test_local_timing_seed42.json').read_text(encoding='utf-8'))
    bundled = dict(methods={}, execution=timing)
    for method in ('joint', 'pcgrad'):
        directory = args.experiment/'results'/f'{method}_step10000_seed42'
        source = directory/'evaluation_test_best_validation.json'
        saved = json.loads(source.read_text(encoding='utf-8'))
        checkpoint = directory/'checkpoints/best_validation.pt'
        assert saved['split'] == 'test' and saved['method'] == method
        assert saved['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        metrics = saved['metrics']
        assert metrics['example_count'] == 10000 and len(metrics['tasks']) == 20
        assert all(sum(t['support']) == 500 for t in metrics['tasks'])
        bundled['methods'][method] = {key: saved[key] for key in
                                      ('step', 'method', 'split', 'checkpoint_sha256', 'metrics')}
        bundled['methods'][method]['evaluation_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        # Keep the deployed evidence independent of local absolute paths.
        bundled['execution']['runs'][method].pop('output', None)
    dest = ROOT/'experiment_data/images/test_results.json'
    dest.write_text(json.dumps(bundled, indent=2)+'\n', encoding='utf-8')
    print('Bundled verified best-validation checkpoint test metrics.')


if __name__ == '__main__':
    main()
