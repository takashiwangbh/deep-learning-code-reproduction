"""DeepAR-like autoregressive Gaussian forecasts with explicit GRU recurrence."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_gate = nn.Linear(input_size, 2 * hidden_size)
        self.hidden_gate = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.input_new = nn.Linear(input_size, hidden_size)
        self.hidden_new = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        xr, xz = self.input_gate(x).chunk(2, -1)
        hr, hz = self.hidden_gate(h).chunk(2, -1)
        r, z = (xr + hr).sigmoid(), (xz + hz).sigmoid()
        candidate = torch.tanh(self.input_new(x) + r * self.hidden_new(h))
        return z * h + (1 - z) * candidate


class DeepAR(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.cell = GRUCell(1, hidden_size)
        self.mean = nn.Linear(hidden_size, 1)
        self.scale = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def step(self, previous: torch.Tensor, hidden: torch.Tensor):
        hidden = self.cell(previous.unsqueeze(-1), hidden)
        return self.mean(hidden).squeeze(-1), F.softplus(self.scale(hidden)).squeeze(-1) + 1e-4, hidden

    def forward(self, series: torch.Tensor):
        """Teacher forcing: predict series[t] from series[:t]. series [B,T]."""
        if series.ndim != 2 or series.shape[1] < 2:
            raise ValueError("series must be [B,T] with T>=2")
        hidden = series.new_zeros(series.shape[0], self.hidden_size)
        means, scales = [], []
        for previous in series[:, :-1].unbind(1):
            mean, scale, hidden = self.step(previous, hidden)
            means.append(mean)
            scales.append(scale)
        return torch.stack(means, 1), torch.stack(scales, 1)

    def negative_log_likelihood(self, series: torch.Tensor, observed: torch.Tensor | None = None):
        mean, scale = self(series)
        loss = 0.5 * math.log(2 * math.pi) + scale.log() + 0.5 * ((series[:, 1:] - mean) / scale).square()
        if observed is None:
            return loss.mean()
        mask = observed[:, 1:].to(loss.dtype)
        return (loss * mask).sum() / mask.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self, history: torch.Tensor, horizon: int):
        hidden = history.new_zeros(history.shape[0], self.hidden_size)
        for previous in history[:, :-1].unbind(1):
            _, _, hidden = self.step(previous, hidden)
        previous = history[:, -1]
        future = []
        for _ in range(horizon):
            mean, scale, hidden = self.step(previous, hidden)
            previous = mean + scale * torch.randn_like(mean)
            future.append(previous)
        return torch.stack(future, 1)


if __name__ == "__main__":
    model = DeepAR()
    data = torch.randn(2, 8)
    loss = model.negative_log_likelihood(data)
    loss.backward()
    print("NLL:", float(loss), "sampled future:", tuple(model.sample(data, 3).shape))
