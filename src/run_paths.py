"""Readable, lossless names for experiment conditions."""


def run_name(weight, method, steps, seed):
    return f"weight{float(weight)!r}_{method}_step{steps}_seed{seed}"
