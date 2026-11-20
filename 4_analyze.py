"""Generate five combined figures from flat result folders, without a GPU.

Usage: python 4_analyze.py --results results
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.results import condition_key, read_runs, write_main_summary


def read_history(folder):
    return [json.loads(line) for line in (folder / "history.jsonl").read_text(encoding="utf-8").splitlines()]


def mean(values):
    return float(np.mean(values))


def sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def plot_all(root, steps=None, output=None):
    root = Path(root)
    runs, preliminary, inventory = read_runs(root)
    if steps is not None:
        runs = [run for run in runs if run["config"]["steps"] == steps]
        preliminary = [run for run in preliminary if run["config"]["steps"] == steps]
        selected = {run["folder"].name for run in runs + preliminary}
        inventory = [{**entry, "status": "excluded_step_budget"}
                     if entry["folder"] not in selected and entry["status"] != "duplicate_control"
                     else entry for entry in inventory]
    if not runs:
        raise ValueError(f"No completed test runs under {root}")
    write_main_summary(root)
    destination = Path(output) if output is not None else root
    destination.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for run in runs:
        run["history"] = read_history(run["folder"])
        grouped[condition_key(run["config"])].append(run)
    groups = dict(sorted(grouped.items()))
    colors = {key: plt.get_cmap("tab20")(i * 2 % 20) for i, key in enumerate(groups)}

    def label(key):
        c = groups[key][0]["config"]
        weight = f" \u03bb={c['weight']:g}" if c["method"] in ("joint", "pcgrad") else "-only"
        name = c['method'].capitalize() + weight
        # The shared budget is documented once; avoid repeating it in every legend.
        if len({r['config']['steps'] for r in runs}) > 1:
            name += f" ({c['steps']} steps)"
        return name

    def values(key, split, metric):
        return [r["summary"][split][metric] for r in groups[key]]

    def save(fig, name):
        fig.savefig(destination / f"summary_{name}.png", dpi=190, bbox_inches="tight")
        plt.close(fig)

    # 1: Learning curves for every method, weight and training budget.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for key, records in groups.items():
        points = defaultdict(list)
        for run in records:
            evaluations = [row for row in run["history"] if "validation" in row]
            for row in evaluations:
                points[row["step"]].append(row["validation"])
            for ax, metric in zip(axes, ("token_loss", "macro_f1")):
                ax.plot([r["step"] for r in evaluations], [r["validation"][metric] for r in evaluations], color=colors[key], alpha=.12)
        steps = sorted(points)
        for ax, metric in zip(axes, ("token_loss", "macro_f1")):
            ax.plot(steps, [mean([r[metric] for r in points[s]]) for s in steps], color=colors[key], lw=2, label=label(key))
    for ax, metric in zip(axes, ("Validation token loss", "Validation macro-F1")):
        ax.set(xlabel="Optimizer steps", ylabel=metric)
        ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=9)
    fig.tight_layout(rect=(0, .14, 1, 1))
    save(fig, "convergence")

    # 2: All full-run trade-offs, with preliminary checks kept separate.
    fig, ax = plt.subplots(figsize=(9, 6))
    for key in groups:
        x, y = values(key, "validation", "token_loss"), values(key, "validation", "macro_f1")
        ax.scatter(x, y, color=colors[key], alpha=.35, s=20)
        ax.errorbar(mean(x), mean(y), xerr=sd(x), yerr=sd(y), fmt="o", capsize=3, color=colors[key], label=label(key))
    for run in preliminary:
        c, s = run["config"], run["summary"]["validation"]
        if c["method"] == "joint" and c.get("train_limit", 0) > 0:
            ax.scatter(s["token_loss"], s["macro_f1"], marker="x", color="gray")
            ax.annotate(f"{c['steps']}-step check, w={c['weight']:g}", (s["token_loss"], s["macro_f1"]), fontsize=7, xytext=(7, -5), textcoords="offset points")
    ax.set(xlabel="Validation token loss (lower is better)", ylabel="Validation macro-F1 (higher is better)", title="Validation task trade-off")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig, "tradeoff")

    summary = {}
    for key, records in groups.items():
        gradients = [row["gradients"] for r in records for row in r["history"] if "gradients" in row]
        info = {"label": label(key), "runs": len(records), "seeds": [r["config"]["seed"] for r in records]}
        for split in ("validation", "test"):
            for metric in ("token_loss", "perplexity", "sentiment_loss", "accuracy", "macro_f1"):
                data = values(key, split, metric)
                info[f"{split}_{metric}_mean"], info[f"{split}_{metric}_sample_sd"] = mean(data), sd(data)
        info["seconds_per_step_mean"] = mean([r["summary"]["elapsed_seconds"] / r["summary"]["steps"] for r in records])
        info["sampled_global_conflict_frequency"] = mean([g["gradient_conflict"] for g in gradients]) if gradients else None
        info["sampled_mean_absolute_cosine"] = mean([abs(g["gradient_cosine"]) for g in gradients]) if gradients else None
        layers = sorted({layer for g in gradients for layer in g["layer_cosines"]}, key=lambda n: (n != "embeddings", n))
        info["layer_conflict_frequencies"] = {layer: mean([g["layer_cosines"][layer] < 0 for g in gradients if layer in g["layer_cosines"]]) for layer in layers}
        summary[key] = info

    # 3: Compare layer-conflict frequencies using one line per condition.
    keys = [k for k in groups if summary[k]["layer_conflict_frequencies"]]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if keys:
        layers = sorted({layer for k in keys for layer in summary[k]["layer_conflict_frequencies"]}, key=lambda n: (n != "embeddings", n))
        for key in keys:
            frequencies = [summary[key]["layer_conflict_frequencies"].get(layer, np.nan)
                           for layer in layers]
            style = "--" if groups[key][0]["config"]["method"] == "pcgrad" else "-"
            ax.plot(range(len(layers)), frequencies, marker="o", markersize=4,
                    linestyle=style, linewidth=1.7, color=colors[key], label=label(key))
        ax.set_xticks(range(len(layers)), layers, rotation=45, ha="right", fontsize=8)
        ax.set(ylabel="Fraction of sampled steps with negative cosine", ylim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.24), ncol=4, fontsize=8)
    else:
        ax.text(.5, .5, "No gradient diagnostics available", ha="center")
        ax.set_axis_off()
    ax.set_title("Pre-projection layer-wise gradient conflict")
    fig.tight_layout(rect=(0, .16, 1, 1))
    save(fig, "layer_conflicts")

    # 4: Timers include validation and exclude setup/final test evaluation.
    keys = list(groups)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(keys)), [summary[k]["seconds_per_step_mean"] for k in keys], color=[colors[k] for k in keys])
    ax.set_xticks(range(len(keys)), [label(k) for k in keys], rotation=35, ha="right", fontsize=8)
    ax.set(ylabel="Recorded training + validation seconds per step", title="Recorded runtime by condition")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    save(fig, "runtime")

    # 5: Every tested weight enters the same overview; match seeds for differences.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    budgets = sorted({r["config"]["steps"] for r in runs})
    weights = sorted({r["config"]["weight"] for r in runs if r["config"]["method"] in ("joint", "pcgrad")})
    for budget in budgets:
        for method, color in (("joint", "#16856e"), ("pcgrad", "#d25b42")):
            chosen = [r for r in runs if r["config"]["method"] == method and r["config"]["steps"] == budget]
            ws = sorted({r["config"]["weight"] for r in chosen})
            for ax, metric, scale in ((axes[0, 0], "macro_f1", 100), (axes[0, 1], "perplexity", 1)):
                data = [[r["summary"]["test"][metric] * scale for r in chosen if r["config"]["weight"] == w] for w in ws]
                if ws:
                    ax.errorbar(ws, [mean(v) for v in data], yerr=[sd(v) for v in data], marker="o", capsize=3, color=color, label=f"{method}, {budget} steps")
        for seed in sorted({r["config"]["seed"] for r in runs}):
            xs, diffs = [], []
            for weight in weights:
                pair = {r["config"]["method"]: r for r in runs if r["config"]["steps"] == budget and r["config"]["seed"] == seed and r["config"]["weight"] == weight and r["config"]["method"] in ("joint", "pcgrad")}
                if len(pair) == 2:
                    xs.append(weight)
                    diffs.append(100 * (pair["pcgrad"]["summary"]["test"]["macro_f1"] - pair["joint"]["summary"]["test"]["macro_f1"]))
            if xs:
                axes[1, 0].plot(xs, diffs, marker="o", label=f"Seed {seed}, {budget} steps")
        for method, metric, ax, scale, color in (("sentiment", "macro_f1", axes[0, 0], 100, "#e28e2c"), ("token", "perplexity", axes[0, 1], 1, "#2879a5")):
            data = [r["summary"]["test"][metric] * scale for r in runs if r["config"]["method"] == method and r["config"]["steps"] == budget]
            if data:
                ax.axhline(mean(data), ls="--", color=color, label=f"{method}-only, {budget} steps")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xscale("log")
        ax.set_xticks(weights, [f"{w:g}" for w in weights])
        ax.set_xlabel("Sentiment weight")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    axes[0, 0].set(title="Sentiment: mean +/- sample SD", ylabel="Test macro-F1 (%)")
    axes[0, 1].set(title="Language modeling: mean +/- sample SD", ylabel="Test perplexity")
    axes[1, 0].axhline(0, color="black", lw=1)
    axes[1, 0].set(title="Paired sentiment differences", ylabel="PCGrad - joint F1 (percentage points)")
    weighted_keys = [k for k in groups if groups[k][0]["config"]["method"] in ("joint", "pcgrad")]
    axes[1, 1].bar(range(len(weighted_keys)), [mean([r["summary"]["elapsed_seconds"] for r in groups[k]]) for k in weighted_keys], color=["#16856e" if groups[k][0]["config"]["method"] == "joint" else "#d25b42" for k in weighted_keys])
    axes[1, 1].set_xticks(range(len(weighted_keys)), [label(k) for k in weighted_keys], rotation=40, ha="right", fontsize=7)
    axes[1, 1].set(title="Recorded training + validation time", ylabel="Seconds per run")
    axes[1, 1].grid(axis="y", alpha=.2)
    if len(weights) == 1 and len(budgets) == 1:
        # A single weight cannot define a weight-response curve. Show the actual
        # per-seed evidence and paired differences instead of a one-point curve.
        for ax in axes.flat:
            ax.clear()
        seeds = sorted({r["config"]["seed"] for r in runs})
        method_colors = {"token": "#2879a5", "sentiment": "#e28e2c",
                         "joint": "#16856e", "pcgrad": "#d25b42"}
        for method, color in method_colors.items():
            chosen = sorted((r for r in runs if r["config"]["method"] == method),
                            key=lambda r: r["config"]["seed"])
            xs = [r["config"]["seed"] for r in chosen]
            if method != "token":
                axes[0, 0].plot(xs, [r["summary"]["test"]["macro_f1"] * 100 for r in chosen],
                                marker="o", color=color, label=method)
            axes[0, 1].plot(xs, [r["summary"]["test"]["perplexity"] for r in chosen],
                            marker="o", color=color, label=method)
        for ax in (axes[0, 0], axes[0, 1]):
            ax.set_xticks(seeds)
            ax.set_xlabel("Random seed")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        axes[0, 0].set(title="Sentiment performance by seed", ylabel="Test macro-F1 (%)")
        axes[0, 1].set(title="Language modeling by seed", ylabel="Test perplexity")
        differences, paired_seeds = [], []
        for seed in seeds:
            pair = {r["config"]["method"]: r for r in runs if r["config"]["seed"] == seed
                    and r["config"]["method"] in ("joint", "pcgrad")}
            if len(pair) == 2:
                paired_seeds.append(seed)
                differences.append(100 * (pair["pcgrad"]["summary"]["test"]["macro_f1"] -
                                          pair["joint"]["summary"]["test"]["macro_f1"]))
        axes[1, 0].bar(range(len(paired_seeds)), differences, color="#d25b42")
        axes[1, 0].set_xticks(range(len(paired_seeds)), [str(seed) for seed in paired_seeds])
        axes[1, 0].axhline(0, color="black", lw=1)
        axes[1, 0].set(title="Paired sentiment differences", xlabel="Random seed",
                        ylabel="PCGrad - joint F1 (percentage points)")
        keys = list(groups)
        times = [[r["summary"]["elapsed_seconds"] / 60 for r in groups[k]] for k in keys]
        axes[1, 1].bar(range(len(keys)), [mean(v) for v in times],
                       yerr=[sd(v) for v in times], capsize=3,
                       color=[method_colors[groups[k][0]["config"]["method"]] for k in keys])
        axes[1, 1].set_xticks(range(len(keys)), [label(k) for k in keys], rotation=25, ha="right")
        axes[1, 1].set(title="Recorded time: mean +/- sample SD", ylabel="Minutes per run")
        for ax in (axes[1, 0], axes[1, 1]):
            ax.grid(axis="y", alpha=.2)
    save(fig, "weight_effects")
    (destination / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (destination / "summary_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(f"Analyzed {len(runs)} unique completed test runs; wrote five summary figures to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--steps", type=int, help="Include only this completed training budget")
    parser.add_argument("--output", type=Path, help="Save filtered figures/summaries separately")
    args = parser.parse_args()
    plot_all(args.results, steps=args.steps, output=args.output)
