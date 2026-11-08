"""Two-task PCGrad and diagnostics on GPT-2's shared backbone."""

import math

import torch


def _dot(left, right):
    # Sum on the device; converting each parameter to a Python float would
    # synchronize CUDA hundreds of times per training step.
    return sum((a.float() * b.float()).sum() for a, b in zip(left, right)).item()


def _norm_sq(grads):
    return _dot(grads, grads)


def cosine(left, right):
    """Return zero when either task has a zero gradient."""
    norm = math.sqrt(_norm_sq(left) * _norm_sq(right))
    return _dot(left, right) / norm if norm else 0.0


def gradient_diagnostics(named_shared, token_grads, sent_grads):
    """Cosines are based on raw (unweighted) gradients by layer."""
    groups = {}
    for (name, _), tok, sent in zip(named_shared, token_grads, sent_grads):
        if name.startswith("h."):
            group = "block_" + name.split(".")[1].zfill(2)
        elif name.startswith("wte.") or name.startswith("wpe."):
            group = "embeddings"
        else:
            group = "final_norm"
        groups.setdefault(group, ([], []))
        groups[group][0].append(tok)
        groups[group][1].append(sent)
    per_layer = {name: cosine(*pair) for name, pair in groups.items()}
    return {
        "gradient_cosine": cosine(token_grads, sent_grads),
        "gradient_conflict": _dot(token_grads, sent_grads) < 0,
        "token_gradient_norm": math.sqrt(_norm_sq(token_grads)),
        "sentiment_gradient_norm": math.sqrt(_norm_sq(sent_grads)),
        "layer_cosines": per_layer,
    }


def combine_shared(token_grads, sent_grads, sentiment_weight, project):
    """Combine weighted task gradients, projecting both against ORIGINAL peers.

    For two tasks the PCGrad random task order has no effect. Coefficients are
    computed from the whole shared backbone, not separately per layer.
    """
    if sentiment_weight <= 0:
        raise ValueError("sentiment_weight must be positive for joint training")
    if not project:
        return [tok + sentiment_weight * sent for tok, sent in zip(token_grads, sent_grads)]
    dot = _dot(token_grads, sent_grads)
    if dot >= 0:
        return [tok + sentiment_weight * sent for tok, sent in zip(token_grads, sent_grads)]
    token_sq = _norm_sq(token_grads)
    sent_sq = _norm_sq(sent_grads)
    if not (token_sq and sent_sq):
        return [tok + sentiment_weight * sent for tok, sent in zip(token_grads, sent_grads)]
    # g_t' = g_t - <g_t,g_s>/||g_s||^2 g_s, and conversely for g_s'.
    # Lambda weights the sentiment task in the final combined update.
    token_coefficient = dot / sent_sq
    sent_coefficient = dot / token_sq
    return [
        (tok - token_coefficient * sent)
        + sentiment_weight * (sent - sent_coefficient * tok)
        for tok, sent in zip(token_grads, sent_grads)
    ]


def assign_joint_gradients(model, token_loss, sentiment_loss, weight, project, measure=True):
    """Get each task's gradients and put the final update in .grad.

    Heads receive their own ordinary gradients; PCGrad touches shared weights
    only. No other task's gradient can leak into a private head.
    """
    shared = list(model.backbone.parameters())
    token_head = list(model.token_head.parameters())
    sent_head = list(model.sentiment_head.parameters())
    tok = torch.autograd.grad(token_loss, shared + token_head, retain_graph=True)
    sent = torch.autograd.grad(sentiment_loss, shared + sent_head)
    token_shared, token_private = tok[:len(shared)], tok[len(shared):]
    sent_shared, sent_private = sent[:len(shared)], sent[len(shared):]
    for param, grad in zip(shared, combine_shared(token_shared, sent_shared, weight, project)):
        param.grad = grad
    for param, grad in zip(token_head, token_private):
        param.grad = grad
    for param, grad in zip(sent_head, sent_private):
        param.grad = weight * grad
    if not measure:
        return None
    diagnostics = gradient_diagnostics(list(model.backbone.named_parameters()), token_shared, sent_shared)
    if project:
        # With exactly two tasks, symmetric PCGrad changes a negative cosine c
        # to -c (unless one vector is zero); nonconflicting vectors stay put.
        diagnostics["post_projection_global_cosine"] = abs(diagnostics["gradient_cosine"])
    return diagnostics
