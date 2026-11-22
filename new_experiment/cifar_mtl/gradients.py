"""Multi-task PCGrad, with projection coefficients computed from a Gram matrix.

G contains original shared task gradients. A coefficient row represents the
current projected gradient as a linear combination of G. The algorithm updates
that row sequentially against original peers, exactly as Algorithm 1 of PCGrad.
This avoids forming a full gradient vector during each of 20*19 projections.
"""
import torch


def project_coefficients(gradients,generator):
    n=len(gradients);gram=gradients@gradients.T
    coefficients=torch.eye(n,device=gradients.device,dtype=gradients.dtype)
    orders=[]
    for i in range(n):
        peers=[j for j in range(n) if j!=i]
        order=[peers[j] for j in torch.randperm(n-1,generator=generator).tolist()]
        orders.append(order)
        for j in order:
            dot=coefficients[i]@gram[:,j]
            # Zero peers produce zero dot and zero correction.
            correction=dot.clamp_max(0)/gram[j,j].clamp_min(1e-20)
            coefficients[i,j]-=correction
    return coefficients,gram,orders


def assign(model,losses,weights,project=False,measure=False,generator=None):
    if not project and not measure:
        (losses*weights).sum().backward()
        return {}
    shared=list(model.shared.parameters());rows=[];private=[]
    for task,loss in enumerate(losses):
        head=list(model.heads[task].parameters())
        grads=torch.autograd.grad(loss,shared+head,retain_graph=task<len(losses)-1)
        rows.append(torch.cat([g.detach().reshape(-1) for g in grads[:len(shared)]]))
        private.append(grads[len(shared):])
    gradients=torch.stack(rows)
    if project:
        coefficients,gram,orders=project_coefficients(gradients,generator)
    else:
        gram=gradients@gradients.T
        coefficients=torch.eye(len(losses),device=gradients.device);orders=[]
    combined=(weights@coefficients)@gradients
    offset=0
    for param in shared:
        param.grad=combined[offset:offset+param.numel()].reshape_as(param)
        offset+=param.numel()
    for task,head in enumerate(model.heads):
        for param,grad in zip(head.parameters(),private[task]):
            param.grad=weights[task]*grad
    if not measure:return {}
    norms=gram.diag().clamp_min(0).sqrt()
    cosine=(gram/(norms[:,None]*norms[None,:]).clamp_min(1e-20)).clamp(-1,1)
    pairs=torch.triu_indices(len(losses),len(losses),1,device=gram.device)
    valid=(norms[pairs[0]]>0)&(norms[pairs[1]]>0)
    pair_cos=cosine[pairs[0],pairs[1]][valid]
    before=weights@gradients
    post=coefficients@gram@coefficients.T
    post_norm=post.diag().clamp_min(0).sqrt()
    post_cos=(post/(post_norm[:,None]*post_norm[None,:]).clamp_min(1e-20)).clamp(-1,1)
    return dict(task_gradient_norms=norms.cpu().tolist(),
                pairwise_cosines=cosine.cpu().tolist(),post_pairwise_cosines=post_cos.cpu().tolist(),
                conflict_fraction=float((pair_cos<0).float().mean()) if len(pair_cos) else 0.,
                mean_absolute_cosine=float(pair_cos.abs().mean()) if len(pair_cos) else 0.,
                projection_relative_norm=float((combined-before).norm()/before.norm().clamp_min(1e-20)),
                projection_orders=orders)
