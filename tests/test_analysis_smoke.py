"""Check that the analysis program consumes saved run data."""

import csv
import json

from importlib import import_module

plot_all = import_module("4_analyze").plot_all


def test_analysis_creates_all_figures(tmp_path):
    main = tmp_path / "main"
    main.mkdir()
    methods = ("token", "sentiment", "joint", "pcgrad")
    for index, method in enumerate(methods):
        folder = main / f"{method}_seed42"
        folder.mkdir()
        row = {
            "step": 2, "elapsed_seconds": 1.0,
            "validation": {"token_loss": 2 + index / 10, "macro_f1": 0.5 + index / 10},
        }
        if method in ("joint", "pcgrad"):
            row["gradients"] = {
                "gradient_conflict": True,
                "token_gradient_norm": 2.0,
                "sentiment_gradient_norm": 1.0,
                "layer_cosines": {"embeddings": -0.2, "block_00": 0.1},
                "post_projection_global_cosine": 0.2,
            }
        (folder / "history.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with (tmp_path / "main_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "steps", "elapsed_seconds",
                                                    "validation_token_loss", "validation_macro_f1"])
        writer.writeheader()
        for index, method in enumerate(methods):
            writer.writerow({"method": method, "steps": 2, "elapsed_seconds": 1,
                             "validation_token_loss": 2 + index / 10,
                             "validation_macro_f1": 0.5 + index / 10})
    plot_all(tmp_path)
    for name in ("convergence", "tradeoff", "layer_conflicts", "runtime"):
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 0
    report = json.loads((tmp_path / "analysis_summary.json").read_text(encoding="utf-8"))
    assert report["pcgrad"]["sampled_global_conflict_frequency"] == 1.0
