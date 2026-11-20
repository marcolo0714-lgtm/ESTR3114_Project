"""Compare every supplied sentiment weight using matched training seeds.

Token-only and sentiment-only controls run once per seed because they do not
use the sentiment weight. Joint and PCGrad run at every weight for every seed.
"""

import argparse
import csv
import math
import json
from pathlib import Path

from importlib import import_module
from src.run_paths import run_name

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


def run_or_reuse(config):
    """Reuse completed matching runs; never overwrite conflicting or partial runs."""
    path = config.output
    if path.exists() and any(path.iterdir()):
        if not all((path / name).is_file() for name in
                   ("config.json", "summary.json", "history.jsonl")):
            raise FileExistsError(f"Incomplete run folder: {path}")
        saved = json.loads((path / "config.json").read_text(encoding="utf-8"))
        keys = ("method", "seed", "steps", "batch_size", "eval_batch_size", "max_length",
                "learning_rate", "weight_decay", "warmup_ratio", "eval_every",
                "diagnostics_every", "train_limit", "validation_limit", "test")
        if config.method in ("joint", "pcgrad"):
            keys += ("weight",)
        if any(saved.get(key) != getattr(config, key) for key in keys):
            raise FileExistsError(f"Different settings already exist at {path}; use --output")
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if summary.get("steps") != config.steps or "test" not in summary:
            raise FileExistsError(f"Incomplete summary at {path}")
        print(f"Reusing {path.name}", flush=True)
        return summary
    return run(config)


def main():
    args = parse_args()
    if any(not math.isfinite(w) or w <= 0 for w in args.weights):
        raise ValueError("weights must be finite and positive")
    # Repeated values should not repeat runs or overwrite their output folders.
    args.weights = list(dict.fromkeys(args.weights))
    args.seeds = list(dict.fromkeys(args.seeds))
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for seed in args.seeds:
        for method in ("token", "sentiment"):
            path = root / "main" / run_name(1.0, method, args.main_steps, seed)
            summaries.append(run_or_reuse(condition(args, method, 1.0, seed, args.main_steps,
                                           path, test=True)))
        for weight in args.weights:
            for method in ("joint", "pcgrad"):
                path = root / "main" / run_name(weight, method, args.main_steps, seed)
                summaries.append(run_or_reuse(condition(args, method, weight, seed, args.main_steps,
                                               path, test=True)))
    fields = ["method", "weight", "seed", "steps", "elapsed_seconds"]
    metrics = ["token_loss", "perplexity", "sentiment_loss", "accuracy", "macro_f1"]
    # Rebuild the table from all completed conditions, including earlier invocations.
    by_condition = {}
    for path in sorted((root / "main").glob("*/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if "test" in item:
            by_condition[run_name(item['weight'], item['method'], item['steps'], item['seed'])] = item
    for item in summaries:
        by_condition[run_name(item['weight'], item['method'], item['steps'], item['seed'])] = item
    with (root / "main_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + [f"{split}_{m}" for split in
                                ("validation", "test") for m in metrics])
        writer.writeheader()
        for item in by_condition.values():
            writer.writerow({**{k: item[k] for k in fields},
                             **{f"{split}_{m}": item[split][m] for split in
                                ("validation", "test") for m in metrics}})
    print(f"Saved {root / 'main_summary.csv'}")


if __name__ == "__main__":
    main()
