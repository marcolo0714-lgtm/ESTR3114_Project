"""Product codebooks trained against detached encoder representations."""
import torch
from torch import nn
from .losses import masked_mean


class ProductQuantizer(nn.Module):
    def __init__(self, hidden_size, segments, size):
        super().__init__()
        self.segments, self.size = segments, size
        d = hidden_size // segments
        self.codes = nn.Parameter(torch.randn(segments, size, d) * 0.02)
        self.transform = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def embeddings(self):
        return self.transform(self.codes)

    def forward(self, pooled, valid):
        target = pooled.detach().reshape(*pooled.shape[:2], self.segments, -1)
        entries = self.embeddings()
        distance = (target.unsqueeze(-2) - entries).square().sum(-1)
        ids = distance.argmin(-1)
        segment = torch.arange(self.segments, device=pooled.device)
        selected = entries[segment, ids]
        loss = masked_mean((selected - target).square().mean((-1, -2)), valid)
        with torch.no_grad():
            counts = torch.stack([torch.bincount(ids[..., s][valid], minlength=self.size)
                                  for s in range(self.segments)])
            probs = counts.float() / counts.sum(-1, keepdim=True).clamp_min(1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
            real = pooled.detach()[valid]
            variance = real.var(0, unbiased=False).mean() if len(real) else pooled.new_tensor(0.)
        return loss, dict(code_counts=counts, codebook_perplexity=entropy.exp().mean(),
                         representation_variance=variance)
