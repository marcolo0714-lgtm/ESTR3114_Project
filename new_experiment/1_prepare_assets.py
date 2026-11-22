"""Download and prepare independent 20-task CIFAR-100 assets."""
import argparse
from pathlib import Path
from cifar_mtl.data import prepare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resources',type=Path,default=Path(__file__).parent/'resources')
    p.add_argument('--validation-per-class',type=int,default=50)
    p.add_argument('--split-seed',type=int,default=123)
    p.add_argument('--archive',type=Path,help='Optional already-downloaded official archive')
    a=p.parse_args()
    prepare(a.resources,a.validation_per_class,a.split_seed,a.archive)


if __name__=='__main__':main()
