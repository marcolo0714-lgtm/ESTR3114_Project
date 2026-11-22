"""Run matched scalar-joint/PCGrad seeds, forwarding training options unchanged."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
from cifar_mtl.results import index_runs

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--methods',nargs='+',choices=['joint','pcgrad'],default=['joint','pcgrad'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--output',type=Path,default=ROOT/'results')
    a,extra=p.parse_known_args()
    if any(x.split('=')[0] in ('--method','--seed') for x in extra):
        p.error('Use --methods and --seeds for the sweep')
    spec=importlib.util.spec_from_file_location('cifar_trainer',ROOT/'2_train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    settings=module.parser().parse_args(extra)
    methods=list(dict.fromkeys(a.methods));seeds=list(dict.fromkeys(a.seeds))
    a.output.mkdir(parents=True,exist_ok=True)
    total=len(methods)*len(seeds);condition=0
    for seed in seeds:
        for method in methods:
            condition+=1
            output=a.output/f'{method}_step{settings.steps}_seed{seed}'
            print(f'Condition {condition}/{total}: {output.name}; evaluation output every {settings.eval_every} steps',flush=True)
            subprocess.run([sys.executable,'-u',str(ROOT/'2_train.py'),*extra,'--method',method,
                            '--seed',str(seed),'--output',str(output)],check=True)
            index_runs(a.output)
    print(f'Completed {total} conditions. Index: {a.output/"main_summary.csv"}',flush=True)


if __name__=='__main__':main()
