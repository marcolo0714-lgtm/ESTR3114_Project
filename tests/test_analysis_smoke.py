"""Check that the analysis program consumes saved run data."""

import csv
import json
import pytest

from importlib import import_module

plot_all = import_module("4_analyze").plot_all


@pytest.mark.parametrize("sweep", [False, True, "named"])
def test_analysis_creates_all_figures(tmp_path, sweep):
    main = tmp_path / "main"
    main.mkdir()
    methods = ("token", "sentiment", "joint", "pcgrad")
    if sweep:
        methods = ("token", "sentiment", "joint_w0.1", "joint_w0.3",
                   "pcgrad_w0.1", "pcgrad_w0.3")
    for index, method in enumerate(methods):
        folder = main / f"{method}_seed42"
        folder.mkdir()
        row = {
            "step": 2, "elapsed_seconds": 1.0,
            "validation": {"token_loss": 2 + index / 10, "macro_f1": 0.5 + index / 10},
        }
        if method.startswith(("joint", "pcgrad")):
            row["gradients"] = {
                "gradient_conflict": True,
                "token_gradient_norm": 2.0,
                "sentiment_gradient_norm": 1.0,
                "layer_cosines": {"embeddings": -0.2, "block_00": 0.1},
                "post_projection_global_cosine": 0.2,
            }
        (folder / "history.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with (tmp_path / "main_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "weight", "steps", "elapsed_seconds",
                                                    "validation_token_loss", "validation_macro_f1"])
        writer.writeheader()
        for index, method in enumerate(methods):
            writer.writerow({"method": method.split("_w")[0],
                             "weight": method.split("_w")[1] if "_w" in method else 1.0,
                             "steps": 2, "elapsed_seconds": 1,
                             "validation_token_loss": 2 + index / 10,
                             "validation_macro_f1": 0.5 + index / 10})
    if sweep == "named":
        from src.run_paths import run_name
        renamed = []
        for method in methods:
            base, _, weight = method.partition("_w")
            name = run_name(float(weight or 1), base, 2, 42)
            (main / f"{method}_seed42").rename(main / name)
            renamed.append(name.rsplit("_seed", 1)[0])
        methods = renamed
    plot_all(tmp_path)
    for name in ("convergence", "tradeoff", "layer_conflicts", "runtime"):
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 0
    report = json.loads((tmp_path / "analysis_summary.json").read_text(encoding="utf-8"))
    assert set(report) == set(methods)
    for method in methods:
        if "pcgrad" in method:
            assert report[method]["sampled_global_conflict_frequency"] == 1.0
    # Distinct input metrics must not be pooled across weights.
    for index, method in enumerate(methods):
        assert report[method]["validation_macro_f1_mean"] == pytest.approx(0.5 + index / 10)
