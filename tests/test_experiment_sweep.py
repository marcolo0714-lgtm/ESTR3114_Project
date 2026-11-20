"""Verify sweep coverage without downloading assets or training GPT-2."""

import csv
import sys
from importlib import import_module

import pytest

experiment = import_module("3_run_experiments")


def test_all_weights_and_matched_seeds_run_once(tmp_path, monkeypatch):
    output = tmp_path / "comparison"
    monkeypatch.setattr(sys, "argv", [
        "3_run_experiments.py", "--weights", "0.1", "0.3", "0.1",
        "--seeds", "42", "43", "42", "--main-steps", "7", "--output", str(output),
    ])
    calls = []

    def fake_run(args):
        calls.append(args)
        metrics = dict.fromkeys(
            ("token_loss", "perplexity", "sentiment_loss", "accuracy", "macro_f1"), 0.5
        )
        return {"method": args.method, "weight": args.weight, "seed": args.seed,
                "steps": args.steps, "elapsed_seconds": 1.0,
                "validation": metrics, "test": metrics}

    monkeypatch.setattr(experiment, "run", fake_run)
    experiment.main()
    assert len(calls) == 12  # (2 controls + 2 methods * 2 weights) * 2 seeds
    for seed in (42, 43):
        for method in ("joint", "pcgrad"):
            assert {c.weight for c in calls if c.seed == seed and c.method == method} == {0.1, 0.3}
        for method in ("token", "sentiment"):
            assert sum(c.seed == seed and c.method == method for c in calls) == 1
    assert len({c.output for c in calls}) == len(calls)
    assert all(c.steps == 7 and c.test for c in calls)
    with (output / "main_summary.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 12
    assert not (output / "pilot_selection.json").exists()


@pytest.mark.parametrize("weight", ["0", "-1", "nan", "inf"])
def test_invalid_weight_fails_before_training(tmp_path, monkeypatch, weight):
    monkeypatch.setattr(sys, "argv", ["run", "--weights", weight, "--output", str(tmp_path)])
    with pytest.raises(ValueError, match="finite and positive"):
        experiment.main()


def test_existing_results_are_preserved(tmp_path, monkeypatch):
    marker = tmp_path / "main_summary.csv"
    marker.write_text("original results")
    monkeypatch.setattr(sys, "argv", ["run", "--output", str(tmp_path)])
    with pytest.raises(FileExistsError, match="not empty"):
        experiment.main()
    assert marker.read_text() == "original results"
