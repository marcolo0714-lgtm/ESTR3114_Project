"""Numerical checks for gradient surgery and private heads."""

import torch
from torch import nn

from src.gradients import assign_joint_gradients, combine_shared, cosine


def test_pcgrad_removes_global_conflict_without_changing_agreeing_gradients():
    a = [torch.tensor([1.0, 0.0])]
    b = [torch.tensor([-1.0, 1.0])]
    combined = combine_shared(a, b, sentiment_weight=1.0, project=True)[0]
    # Both projected task gradients have nonnegative mutual cosine.
    projected_a = a[0] - (torch.dot(a[0], b[0]) / torch.dot(b[0], b[0])) * b[0]
    projected_b = b[0] - (torch.dot(a[0], b[0]) / torch.dot(a[0], a[0])) * a[0]
    assert torch.allclose(combined, projected_a + projected_b)
    assert cosine([projected_a], [projected_b]) >= 0
    assert torch.allclose(combine_shared(a, b, 2, False)[0], a[0] + 2 * b[0])
    assert torch.allclose(combine_shared(a, [torch.tensor([2.0, 1.0])], 2, True)[0],
                          a[0] + 2 * torch.tensor([2.0, 1.0]))


class TinyJoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2, bias=False)
        self.token_head = nn.Linear(2, 1, bias=False)
        self.sentiment_head = nn.Linear(2, 1, bias=False)


def test_heads_get_only_own_weighted_task_gradient():
    model = TinyJoint()
    with torch.no_grad():
        model.backbone.weight.copy_(torch.eye(2))
        model.token_head.weight.copy_(torch.tensor([[1.0, 2.0]]))
        model.sentiment_head.weight.copy_(torch.tensor([[3.0, 4.0]]))
    x = torch.tensor([[2.0, 3.0]])
    features = model.backbone(x)
    tok = model.token_head(features).sum()
    sent = model.sentiment_head(features).sum()
    expected_tok_head = torch.autograd.grad(tok, model.token_head.weight, retain_graph=True)[0]
    expected_sent_head = torch.autograd.grad(sent, model.sentiment_head.weight, retain_graph=True)[0]
    diag = assign_joint_gradients(model, tok, sent, weight=0.4, project=True)
    assert torch.allclose(model.token_head.weight.grad, expected_tok_head)
    assert torch.allclose(model.sentiment_head.weight.grad, 0.4 * expected_sent_head)
    assert "gradient_cosine" in diag
