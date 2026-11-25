"""向量版 RealNVP：交替掩码仿射耦合与精确 log determinant。"""

import math

import torch
from torch import nn


class AffineCoupling(nn.Module):
    def __init__(self, dim, mask, hidden=32):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.network = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2 * dim))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def parameters_for(self, x):
        scale, shift = self.network(x * self.mask).chunk(2, -1)
        free = 1 - self.mask
        return 0.8 * torch.tanh(scale) * free, shift * free

    def forward(self, x):
        scale, shift = self.parameters_for(x)
        z = x * self.mask + (1 - self.mask) * (x * scale.exp() + shift)
        return z, scale.sum(-1)

    def inverse(self, z):
        scale, shift = self.parameters_for(z)  # 固定的那半向量在正反变换中完全相同。
        return z * self.mask + (1 - self.mask) * ((z - shift) * (-scale).exp())


class RealNVP(nn.Module):
    def __init__(self, dim=4, layers=4):
        super().__init__()
        assert dim >= 2 and layers >= 1
        self.dim = dim
        self.layers = nn.ModuleList(
            AffineCoupling(dim, torch.tensor([(i + layer) % 2 for i in range(dim)]))
            for layer in range(layers)
        )

    def forward(self, x):
        total_logdet = x.new_zeros(x.shape[0])
        for layer in self.layers:
            x, logdet = layer(x)
            total_logdet = total_logdet + logdet
        return x, total_logdet

    def inverse(self, z):
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z

    def log_prob(self, x):
        z, logdet = self(x)
        log_base = -0.5 * (z.square() + math.log(2 * math.pi)).sum(-1)
        return log_base + logdet

    @torch.no_grad()
    def sample(self, count):
        return self.inverse(torch.randn(count, self.dim, device=next(self.parameters()).device))


if __name__ == "__main__":
    torch.manual_seed(0)
    model = RealNVP()
    x = torch.randn(3, 4)
    z, logdet = model(x)
    assert torch.allclose(model.inverse(z), x, atol=1e-5)
    loss = -model.log_prob(x).mean()
    loss.backward()
    print("latent", tuple(z.shape), "logdet", tuple(logdet.shape), "NLL", round(loss.item(), 3))
