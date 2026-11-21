"""Project only the intersection of token/concept gradient supports."""
import math
import torch


def combine(token, concept, weight, project):
    shared = [i for i, (a, b) in enumerate(zip(token, concept)) if a is not None and b is not None]
    if shared:
        dot = sum((token[i].float() * concept[i].float()).sum() for i in shared)
        nt = sum(token[i].float().square().sum() for i in shared)
        nc = sum(concept[i].float().square().sum() for i in shared)
        cosine = (dot / (nt * nc).sqrt().clamp_min(1e-30)).item()
        conflict = dot.item() < 0 and nt.item() > 0 and nc.item() > 0
    else:
        dot = nt = nc = torch.tensor(0.)
        cosine, conflict = 0., False
    result, delta_sq, base_sq = [], 0., 0.
    shared_set = set(shared)
    for i, (a, b) in enumerate(zip(token, concept)):
        base = (a if a is not None else 0) + (weight*b if b is not None else 0)
        if a is None and b is None:
            result.append(None)
            continue
        update = base
        if project and conflict and i in shared_set:
            update = a - (dot/nc).to(a.dtype)*b + weight*(b-(dot/nt).to(b.dtype)*a)
        delta_sq = delta_sq + (update-base).float().square().sum()
        base_sq = base_sq + base.float().square().sum()
        result.append(update)
    diagnostics = dict(gradient_cosine=cosine, gradient_conflict=conflict,
                       token_gradient_norm=math.sqrt(nt.item()),
                       concept_gradient_norm=math.sqrt(nc.item()),
                       shared_parameter_tensors=len(shared),
                       projection_relative_norm=float((delta_sq/(base_sq+1e-30)).sqrt())
                       if torch.is_tensor(base_sq) else 0.)
    return result, diagnostics


def assign_gradients(model, output, weight, beta, project=False, measure=False):
    params = [p for p in model.parameters() if p.requires_grad]
    losses = output['token_loss'], output['concept_loss'], output['vq_loss']
    if not project and not measure:
        (losses[0] + weight*losses[1] + beta*losses[2]).backward()
        return {}
    token = torch.autograd.grad(losses[0], params, retain_graph=True, allow_unused=True)
    concept = torch.autograd.grad(losses[1], params, retain_graph=True, allow_unused=True)
    vq = torch.autograd.grad(losses[2], params, allow_unused=True)
    grads, diagnostics = combine(token, concept, weight, project)
    for p, grad, q in zip(params, grads, vq):
        p.grad = grad if q is None or beta == 0 else beta*q if grad is None else grad+beta*q
    return diagnostics
