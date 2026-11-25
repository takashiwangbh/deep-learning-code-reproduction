"""单尺度 Glow：squeeze、ActNorm、可逆 1×1 卷积与仿射耦合。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def squeeze2d(x):
    batch, channels, height, width = x.shape
    assert height % 2 == width % 2 == 0
    return x.reshape(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(batch, channels * 4, height // 2, width // 2)


def unsqueeze2d(x):
    batch, channels, height, width = x.shape
    assert channels % 4 == 0
    return x.reshape(batch, channels // 4, 2, 2, height, width).permute(0, 1, 4, 2, 5, 3).reshape(batch, channels // 4, height * 2, width * 2)


class ActNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, x):
        if not bool(self.initialized):
            with torch.no_grad():
                self.bias.copy_(-x.mean((0, 2, 3), keepdim=True))
                self.log_scale.copy_(-torch.log((x + self.bias).std((0, 2, 3), keepdim=True, unbiased=False) + 1e-6))
                self.initialized.fill_(True)
        y = (x + self.bias) * self.log_scale.exp()
        logdet = x.shape[2] * x.shape[3] * self.log_scale.sum()
        return y, logdet.expand(x.shape[0])

    def inverse(self, y):
        if not bool(self.initialized):
            raise RuntimeError("ActNorm needs one forward pass for data-dependent initialization")
        return y * (-self.log_scale).exp() - self.bias


class Invertible1x1Conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(channels, channels))
        self.weight = nn.Parameter(q)

    def forward(self, x):
        y = F.conv2d(x, self.weight[:, :, None, None])
        _, logabsdet = torch.linalg.slogdet(self.weight)
        return y, (x.shape[2] * x.shape[3] * logabsdet).expand(x.shape[0])

    def inverse(self, y):
        return F.conv2d(y, torch.linalg.inv(self.weight)[:, :, None, None])


class AffineCoupling(nn.Module):
    def __init__(self, channels, hidden=32):
        super().__init__()
        assert channels % 2 == 0
        self.network = nn.Sequential(nn.Conv2d(channels // 2, hidden, 3, padding=1), nn.ReLU(), nn.Conv2d(hidden, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, channels, 3, padding=1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def coupling_parameters(self, left):
        log_scale, shift = self.network(left).chunk(2, dim=1)
        return log_scale.tanh(), shift

    def forward(self, x):
        left, right = x.chunk(2, dim=1)
        log_scale, shift = self.coupling_parameters(left)
        return torch.cat((left, right * log_scale.exp() + shift), dim=1), log_scale.flatten(1).sum(-1)

    def inverse(self, y):
        left, right = y.chunk(2, dim=1)
        log_scale, shift = self.coupling_parameters(left)
        return torch.cat((left, (right - shift) * (-log_scale).exp()), dim=1)


class GlowStep(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = ActNorm(channels)
        self.mix = Invertible1x1Conv(channels)
        self.coupling = AffineCoupling(channels)

    def forward(self, x):
        x, a = self.norm(x)
        x, b = self.mix(x)
        x, c = self.coupling(x)
        return x, a + b + c

    def inverse(self, z):
        return self.norm.inverse(self.mix.inverse(self.coupling.inverse(z)))


class Glow(nn.Module):
    def __init__(self, channels=1, steps=2):
        super().__init__()
        self.channels = channels
        self.steps = nn.ModuleList(GlowStep(channels * 4) for _ in range(steps))

    def forward(self, x):
        assert x.shape[1] == self.channels
        z = squeeze2d(x)
        logdet = x.new_zeros(x.shape[0])
        for step in self.steps:
            z, delta = step(z)
            logdet = logdet + delta
        return z, logdet

    def inverse(self, z):
        for step in reversed(self.steps):
            z = step.inverse(z)
        return unsqueeze2d(z)

    def log_prob(self, x):
        z, logdet = self(x)
        return -0.5 * (z.square() + math.log(2 * math.pi)).flatten(1).sum(-1) + logdet


if __name__ == "__main__":
    torch.manual_seed(0)
    model = Glow()
    x = torch.randn(2, 1, 8, 8)
    z, logdet = model(x)
    assert torch.allclose(model.inverse(z), x, atol=1e-5)
    loss = -model.log_prob(x).mean()
    loss.backward()
    print("latent", tuple(z.shape), "logdet", tuple(logdet.shape), "NLL", round(loss.item(), 3))
