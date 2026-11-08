"""Build trade-off, learning, gradient-conflict and overhead figures.

Usage: python analyze.py --results results/comparison
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_history(folder):
    return [json.loads(line) for line in (folder / "history.jsonl").read_text(
        encoding="utf-8").splitlines()]


def plot_all(root):
    main = root / "main"
    folders = sorted(p for p in main.iterdir() if (p / "history.jsonl").exists())
    if not folders:
        raise ValueError(f"No run histories in {main}")
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    histories = {p.name: read_history(p) for p in folders}

    # Curves: all seeds faint, average at each evaluation step bold.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"token": "#1f77b4", "sentiment": "#ff7f0e",
              "joint": "#2ca02c", "pcgrad": "#d62728"}
    for method, color in colors.items():
        by_step = defaultdict(list)
        for name, history in histories.items():
            if not name.startswith(method + "_seed"):
                continue
            points = [row for row in history if "validation" in row]
            for key, metric in enumerate(("token_loss", "macro_f1")):
                axes[key].plot([r["step"] for r in points],
                               [r["validation"][metric] for r in points],
                               color=color, alpha=0.18)
            for row in points:
                by_step[row["step"]].append(row["validation"])
        for key, metric in enumerate(("token_loss", "macro_f1")):
            steps = sorted(by_step)
            axes[key].plot(steps, [np.mean([r[metric] for r in by_step[s]]) for s in steps],
                           color=color, lw=2.4, label=method)
    for ax, label in zip(axes, ("Validation token loss", "Validation macro-F1")):
        ax.set(xlabel="Optimizer steps", ylabel=label)
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figures / "convergence.png", dpi=170)
    plt.close(fig)

    # Final validation trade-off includes the separately tuned pilot points.
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, color in colors.items():
        vals = [h[-1]["validation"] for n, h in histories.items()
                if n.startswith(method + "_seed")]
        if vals:
            x = [v["token_loss"] for v in vals]
            y = [v["macro_f1"] for v in vals]
            ax.scatter(x, y, color=color, alpha=0.55)
            ax.errorbar(np.mean(x), np.mean(y), xerr=np.std(x), yerr=np.std(y),
                        fmt="o", capsize=3, color=color, label=method)
    pilot = root / "pilot_selection.json"
    if pilot.exists():
        for row in json.loads(pilot.read_text(encoding="utf-8"))["pilot_candidates"]:
            val = row["validation"]
            ax.annotate(f"pilot λ={row['weight']:g}",
                        (val["token_loss"], val["macro_f1"]), fontsize=8)
            ax.scatter(val["token_loss"], val["macro_f1"], marker="x", color="gray")
    ax.set(xlabel="Validation token loss (lower is better)",
           ylabel="Sentiment macro-F1 (higher is better)", title="Empirical trade-off")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "tradeoff.png", dpi=170)
    plt.close(fig)

    # Diagnostic rates are PRE-projection, sampled on the same cadence.
    group_values = defaultdict(lambda: defaultdict(list))
    norm_ratios = defaultdict(list)
    global_conflicts = defaultdict(list)
    post_cosines = defaultdict(list)
    for name, history in histories.items():
        method = name.split("_seed")[0]
        for row in history:
            if "gradients" not in row:
                continue
            grad = row["gradients"]
            global_conflicts[method].append(grad["gradient_conflict"])
            if "post_projection_global_cosine" in grad:
                post_cosines[method].append(grad["post_projection_global_cosine"])
            for group, value in grad["layer_cosines"].items():
                group_values[method][group].append(value < 0)
            if grad["sentiment_gradient_norm"]:
                norm_ratios[method].append(grad["token_gradient_norm"] /
                                           grad["sentiment_gradient_norm"])
    if group_values:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        groups = sorted({g for method in group_values.values() for g in method},
                        key=lambda g: (g != "embeddings", g))
        methods = [m for m in ("joint", "pcgrad") if m in group_values]
        for index, method in enumerate(methods):
            values = [np.mean(group_values[method][g]) if group_values[method][g]
                      else np.nan for g in groups]
            width = 0.8 / len(methods)
            ax.bar(np.arange(len(groups)) - 0.4 + (index + .5) * width,
                   values, width, label=method, color=colors[method])
        ax.set_xticks(range(len(groups)), groups, rotation=45, ha="right")
        ax.set(ylabel="Fraction of sampled steps with negative cosine", ylim=(0, 1),
               title="Pre-update layer-wise gradient conflict")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "layer_conflicts.png", dpi=170)
        plt.close(fig)

    with (root / "main_summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = {}
    for method in colors:
        selected = [row for row in rows if row["method"] == method]
        if selected:
            summary[method] = {
                "seconds_per_step_mean": float(np.mean([float(r["elapsed_seconds"]) /
                                                     float(r["steps"]) for r in selected])),
                "validation_token_loss_mean": float(np.mean([float(r["validation_token_loss"]) for r in selected])),
                "validation_macro_f1_mean": float(np.mean([float(r["validation_macro_f1"]) for r in selected])),
                "sampled_token_to_sentiment_gradient_norm_ratio_mean": (
                    float(np.mean(norm_ratios[method])) if norm_ratios[method] else None
                ),
                "sampled_global_conflict_frequency": (
                    float(np.mean(global_conflicts[method])) if global_conflicts[method] else None
                ),
                "sampled_post_projection_global_cosine_mean": (
                    float(np.mean(post_cosines[method])) if post_cosines[method] else None
                ),
            }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = list(summary)
    ax.bar(methods, [summary[m]["seconds_per_step_mean"] for m in methods],
           color=[colors[m] for m in methods])
    ax.set(ylabel="Mean wall-clock seconds per step", title="Training and evaluation overhead")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "runtime.png", dpi=170)
    plt.close(fig)
    (root / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Saved figures and", root / "analysis_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/comparison"))
    plot_all(parser.parse_args().results)
