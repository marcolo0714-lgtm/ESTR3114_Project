"""Run a rectangular weight/beta grid with single-task controls once per seed."""
import argparse
import subprocess
import sys
from pathlib import Path
from src.results import index_runs, number, run_name

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', nargs='+', type=float, default=[.3, 1., 3.])
    p.add_argument('--betas', nargs='+', type=float, default=[.1])
    p.add_argument('--seeds', nargs='+', type=int, default=[42,43,44])
    p.add_argument('--output', type=Path, default=ROOT/'results')
    # Remaining flags are forwarded to 2_train.py for one source of defaults.
    a, extra = p.parse_known_args()
    forbidden = ('--method', '--weight', '--lambda', '--beta', '--seed', '--output')
    if any(x.split('=')[0] in forbidden for x in extra):
        p.error('Use sweep arguments for method/weight/beta/seed/output')
    import importlib.util
    spec = importlib.util.spec_from_file_location('ncp_train', ROOT/'2_train.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    settings = module.parser().parse_args(extra)
    for x in a.weights+a.betas:
        number(x)
        if x <= 0:
            p.error('Sweep weights and betas must be positive')
    a.output.mkdir(parents=True, exist_ok=True)
    for seed in dict.fromkeys(a.seeds):
        conditions = [('token_baseline',0.,0.), ('concept_token_only',0.,0.)]
        conditions += [(m,w,b) for w in dict.fromkeys(a.weights) for b in dict.fromkeys(a.betas)
                       for m in ('joint','pcgrad')]
        for method, weight, beta in conditions:
            path = a.output/run_name(method, weight, beta, settings.steps, seed)
            command = [sys.executable, str(ROOT/'2_train.py'), *extra, '--method',method,
                       '--weight',str(weight),'--beta',str(beta),'--seed',str(seed),'--output',str(path)]
            subprocess.run(command, check=True)
            index_runs(a.output)


if __name__ == '__main__':
    main()
