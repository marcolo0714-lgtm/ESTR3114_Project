"""Verify sweep coverage without downloading assets or training GPT-2."""

import csv
import json
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
    assert calls[0].output.name == "weight1.0_token_step7_seed42"
    assert all(c.output.parent == output for c in calls)
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
    path = tmp_path / "weight1.0_token_step1000_seed42"
    path.mkdir()
    marker = path / "history.jsonl"
    marker.write_text("original results")
    from argparse import Namespace
    with pytest.raises(FileExistsError, match="Incomplete"):
        experiment.run_or_reuse(Namespace(output=path))
    assert marker.read_text() == "original results"


def test_repeated_invocations_reuse_controls_and_keep_summary(tmp_path, monkeypatch):
    calls = []

    def fake_run(config):
        calls.append(config)
        config.output.mkdir(parents=True)
        (config.output / "config.json").write_text(json.dumps(vars(config), default=str))
        metrics = dict.fromkeys(
            ("token_loss", "perplexity", "sentiment_loss", "accuracy", "macro_f1"), 0.5)
        summary = {"method": config.method, "weight": config.weight, "seed": config.seed,
                   "steps": config.steps, "elapsed_seconds": 1,
                   "validation": metrics, "test": metrics}
        (config.output / "summary.json").write_text(json.dumps(summary))
        (config.output / "history.jsonl").write_text(json.dumps({'step': config.steps}) + '\n')
        return summary

    monkeypatch.setattr(experiment, "run", fake_run)
    def invoke(weight, steps="7", extra=()):
        monkeypatch.setattr(sys, "argv", ["run", "--weights", weight, "--seeds", "42",
                            "--main-steps", steps, "--output", str(tmp_path), *extra])
        experiment.main()

    invoke("0.1")
    invoke("0.3")
    assert len(calls) == 6  # controls reused; two new weighted conditions
    invoke("0.3")
    assert len(calls) == 6
    invoke("0.3", steps="8")
    assert len(calls) == 10  # new step budget gets separate folders
    with (tmp_path / "main_summary.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 10
    with pytest.raises(FileExistsError, match="Different settings"):
        invoke("0.3", extra=("--batch-size", "4"))
