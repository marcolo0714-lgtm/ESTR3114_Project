"""Compare every supplied sentiment weight using matched training seeds.

Token-only and sentiment-only controls run once per seed because they do not
use the sentiment weight. Joint and PCGrad run at every weight for every seed.
"""

import argparse
import csv
import math
from pathlib import Path

from importlib import import_module

# importlib supports the numbered training script filename.
train = import_module("2_train")
ROOT, run = train.ROOT, train.run


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--main-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--diagnostics-every", type=int, default=25)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "comparison")
    return parser.parse_args()


def condition(args, method, weight, seed, steps, path, train_limit=0, validation_limit=0, test=False):
    # Construct the same Namespace accepted by train.run. Each condition starts
    # from pretrained weights and uses the same seed and batch shuffle.
    return argparse.Namespace(
        method=method, weight=weight, seed=seed, steps=steps,
        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        max_length=args.max_length, learning_rate=args.learning_rate,
        weight_decay=0.01, warmup_ratio=0.1,
        eval_every=args.eval_every, diagnostics_every=args.diagnostics_every,
        train_limit=train_limit, validation_limit=validation_limit,
        test=test, output=path, device=args.device, save_model=False,
    )


def weight_name(weight):
    """Keep enough digits to give distinct floating-point weights distinct paths."""
    return repr(weight)


def main():
    args = parse_args()
    if any(not math.isfinite(w) or w <= 0 for w in args.weights):
        raise ValueError("weights must be finite and positive")
    # Repeated values should not repeat runs or overwrite their output folders.
    args.weights = list(dict.fromkeys(args.weights))
    args.seeds = list(dict.fromkeys(args.seeds))
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {root}; choose a new --output")
    root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for seed in args.seeds:
        for method in ("token", "sentiment"):
            path = root / "main" / f"{method}_seed{seed}"
            summaries.append(run(condition(args, method, 1.0, seed, args.main_steps,
                                           path, test=True)))
        for weight in args.weights:
            for method in ("joint", "pcgrad"):
                path = root / "main" / f"{method}_w{weight_name(weight)}_seed{seed}"
                summaries.append(run(condition(args, method, weight, seed, args.main_steps,
                                               path, test=True)))
    fields = ["method", "weight", "seed", "steps", "elapsed_seconds"]
    metrics = ["token_loss", "perplexity", "sentiment_loss", "accuracy", "macro_f1"]
    with (root / "main_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + [f"{split}_{m}" for split in
                                ("validation", "test") for m in metrics])
        writer.writeheader()
        for item in summaries:
            writer.writerow({**{k: item[k] for k in fields},
                             **{f"{split}_{m}": item[split][m] for split in
                                ("validation", "test") for m in metrics}})
    print(f"Saved {root / 'main_summary.csv'}")


if __name__ == "__main__":
    main()
