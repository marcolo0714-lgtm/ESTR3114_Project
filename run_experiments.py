"""Pilot lambda on validation, then run four methods with matched seeds.

The test split is evaluated only for final main runs, never for lambda choice.
"""

import argparse
import csv
import json
from pathlib import Path

from train import ROOT, run


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--pilot-steps", type=int, default=100)
    parser.add_argument("--main-steps", type=int, default=1000)
    parser.add_argument("--pilot-train-limit", type=int, default=2000)
    parser.add_argument("--pilot-validation-limit", type=int, default=500)
    parser.add_argument("--max-token-loss-gap", type=float, default=0.05,
                        help="Allow this fractional token-loss increase over token-only pilot")
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


def choose_weight(control, candidates, gap):
    """Best macro-F1 under a token-loss guardrail; fallback: best token loss."""
    bound = control["validation"]["token_loss"] * (1 + gap)
    eligible = [run for run in candidates if run["validation"]["token_loss"] <= bound]
    if eligible:
        return max(eligible, key=lambda r: (r["validation"]["macro_f1"],
                                            -r["validation"]["token_loss"]))["weight"]
    return min(candidates, key=lambda r: (r["validation"]["token_loss"],
                                          -r["validation"]["macro_f1"]))["weight"]


def main():
    args = parse_args()
    if any(w <= 0 for w in args.weights) or args.max_token_loss_gap < 0:
        raise ValueError("weights must be positive; token-loss gap must be nonnegative")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    pilot_seed = args.seeds[0]
    pilot = root / "pilot"
    baseline = run(condition(args, "token", 1.0, pilot_seed, args.pilot_steps,
                             pilot / "token", args.pilot_train_limit, args.pilot_validation_limit))
    candidates = [run(condition(args, "joint", w, pilot_seed, args.pilot_steps,
                                pilot / f"joint_w{w:g}", args.pilot_train_limit,
                                args.pilot_validation_limit)) for w in args.weights]
    weight = choose_weight(baseline, candidates, args.max_token_loss_gap)
    selection = {"selected_weight": weight, "max_token_loss_gap": args.max_token_loss_gap,
                 "pilot_control": baseline, "pilot_candidates": candidates}
    (root / "pilot_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(f"Selected lambda={weight:g} from validation pilot", flush=True)

    summaries = []
    for seed in args.seeds:
        for method in ("token", "sentiment", "joint", "pcgrad"):
            path = root / "main" / f"{method}_seed{seed}"
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
