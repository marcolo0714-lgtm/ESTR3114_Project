"""Masked reductions: padding contributes neither loss nor denominator."""
def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)
