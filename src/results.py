"""Read flat result folders and index completed runs without counting controls twice."""

import csv
import json
from pathlib import Path

from .run_paths import run_name


def condition_key(config):
    """Single-task updates do not use lambda; keep their controls in one group."""
    method = config["method"]
    weight = f"_w{float(config['weight'])!r}" if method in ("joint", "pcgrad") else ""
    return f"{method}{weight}_step{config['steps']}"


def read_runs(root):
    """Return canonical completed test runs, validation-only runs, and an inventory.

    Older controls with different placeholder weights remain on disk. Reuse
    their metrics only once if settings, seed, and evaluation results match.
    """
    root = Path(root)
    candidates, validation_only, inventory = [], [], []
    folders = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("weight"))
    for folder in folders:
        entry = {"folder": folder.name, "status": "incomplete"}
        inventory.append(entry)
        if not all((folder / name).exists() for name in
                   ("config.json", "summary.json", "history.jsonl")):
            continue
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        lines = (folder / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if not lines or json.loads(lines[-1]).get("step") != config["steps"]:
            continue
        record = {"folder": folder, "config": config, "summary": summary}
        if "test" not in summary:
            entry["status"] = "validation_only"
            validation_only.append(record)
            continue
        candidates.append((record, entry))

    unique = {}
    # Prefer the standard 1.0 placeholder when equivalent controls are present.
    candidates.sort(key=lambda pair: (float(pair[0]["config"]["weight"]) != 1.0,
                                     pair[0]["folder"].name))
    for record, entry in candidates:
        config, summary = record["config"], record["summary"]
        settings = {k: v for k, v in config.items() if k in (
            "method", "weight", "seed", "steps", "batch_size", "eval_batch_size",
            "max_length", "learning_rate", "weight_decay", "warmup_ratio",
            "eval_every", "diagnostics_every", "train_limit", "validation_limit")}
        if config["method"] in ("token", "sentiment"):
            settings.pop("weight", None)
        settings["assets"] = {k: v for k, v in config.get("assets", {}).items() if k in (
            "model_revision_resolved", "dataset_revision_resolved", "dataset_fingerprints")}
        key = json.dumps(settings, sort_keys=True)
        if key in unique:
            previous = unique[key]
            if any(previous["summary"][split] != summary[split] for split in ("validation", "test")):
                raise ValueError(f"Conflicting repeated run: {record['folder']}")
            entry.update(status="duplicate_control", canonical_folder=previous["folder"].name)
        else:
            entry["status"] = "included"
            unique[key] = record
    return sorted(unique.values(), key=lambda r: r["folder"].name), validation_only, inventory


def write_main_summary(root, extra=()):
    """Rebuild the CSV from completed flat runs, including interrupted sweeps."""
    root = Path(root)
    runs, _, _ = read_runs(root)
    items = {run_name(r["summary"]["weight"], r["summary"]["method"],
                      r["summary"]["steps"], r["summary"]["seed"]): r["summary"] for r in runs}
    for item in extra:
        items[run_name(item["weight"], item["method"], item["steps"], item["seed"])] = item
    fields = ["method", "weight", "seed", "steps", "elapsed_seconds"]
    metrics = ["token_loss", "perplexity", "sentiment_loss", "accuracy", "macro_f1"]
    with (root / "main_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + [f"{split}_{m}" for split in
                                ("validation", "test") for m in metrics])
        writer.writeheader()
        for item in items.values():
            writer.writerow({**{k: item[k] for k in fields},
                             **{f"{split}_{m}": item[split][m] for split in
                                ("validation", "test") for m in metrics}})
    return root / "main_summary.csv"
