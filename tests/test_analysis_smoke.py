"""Verify combined analysis, duplicate controls and recovery without a CSV."""
import csv
import json
from importlib import import_module

import pytest
from src.results import read_runs
from src.run_paths import run_name

plot_all = import_module("4_analyze").plot_all


def save_run(root, method, weight, score, test=True, steps=2):
    folder = root / run_name(weight, method, steps, 42)
    folder.mkdir()
    config = {"method": method, "weight": weight, "seed": 42, "steps": steps,
              "train_limit": 8 if not test else 0}
    metrics = {"token_loss": 2.0, "perplexity": 7.4, "sentiment_loss": 1.0,
               "accuracy": score, "macro_f1": score, "examples": 8}
    summary = {**config, "elapsed_seconds": 1.0, "validation": metrics}
    if test:
        summary["test"] = metrics
    row = {"step": steps, "validation": metrics}
    if method in ("joint", "pcgrad"):
        row["gradients"] = {"gradient_conflict": True, "gradient_cosine": -0.2,
            "token_gradient_norm": 2.0, "sentiment_gradient_norm": 1.0,
            "layer_cosines": {"embeddings": -0.2, "block_00": 0.1}}
    for name, value in (("config.json", config), ("summary.json", summary)):
        (folder / name).write_text(json.dumps(value), encoding="utf-8")
    (folder / "history.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return folder


@pytest.mark.parametrize("weights", [[0.3], [0.1, 0.3, 1.0, 3.0]])
def test_flat_analysis_combines_weights_and_deduplicates_controls(tmp_path, weights):
    save_run(tmp_path, "token", 1.0, 0.2)
    save_run(tmp_path, "token", 0.3, 0.2)  # Same seed/metrics, unused placeholder.
    save_run(tmp_path, "sentiment", 1.0, 0.7)
    expected = {"token_step2": 0.2, "sentiment_step2": 0.7}
    for index, weight in enumerate(weights):
        for method in ("joint", "pcgrad"):
            score = 0.3 + index / 10
            save_run(tmp_path, method, weight, score)
            expected[f"{method}_w{weight!r}_step2"] = score
    save_run(tmp_path, "joint", 0.1, 0.99, test=False, steps=100)
    # No CSV exists: analysis must rebuild it from individual summaries.
    plot_all(tmp_path)
    for name in ("convergence", "tradeoff", "layer_conflicts", "runtime", "weight_effects"):
        assert (tmp_path / f"summary_{name}.png").stat().st_size > 0
    assert not (tmp_path / "figures").exists()
    report = json.loads((tmp_path / "analysis_summary.json").read_text())
    assert set(report) == set(expected)
    for key, score in expected.items():
        assert report[key]["test_macro_f1_mean"] == pytest.approx(score)
        assert report[key]["runs"] == 1
        if key.startswith("pcgrad"):
            assert report[key]["sampled_global_conflict_frequency"] == 1.0
    with (tmp_path / "main_summary.csv").open(newline="") as f:
        assert len(list(csv.DictReader(f))) == len(expected)
    inventory = json.loads((tmp_path / "summary_inventory.json").read_text())
    assert sum(row["status"] == "duplicate_control" for row in inventory) == 1
    assert sum(row["status"] == "validation_only" for row in inventory) == 1


def test_conflicting_control_replicates_are_not_silently_pooled(tmp_path):
    save_run(tmp_path, "token", 1.0, 0.2)
    save_run(tmp_path, "token", 0.3, 0.4)
    with pytest.raises(ValueError, match="Conflicting repeated run"):
        read_runs(tmp_path)
