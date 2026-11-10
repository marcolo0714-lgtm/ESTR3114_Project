from importlib import import_module

choose_weight = import_module("3_run_experiments").choose_weight


def test_validation_selection_respects_token_loss_guardrail():
    baseline = {"validation": {"token_loss": 2.0}}
    candidates = [
        {"weight": 0.1, "validation": {"token_loss": 2.05, "macro_f1": 0.6}},
        {"weight": 1.0, "validation": {"token_loss": 2.5, "macro_f1": 0.8}},
    ]
    assert choose_weight(baseline, candidates, gap=0.05) == 0.1
