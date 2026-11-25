"""Generic N-BEATS residual backcast/forecast stacks."""

import torch
from torch import nn


class NBeatsBlock(nn.Module):
    def __init__(self, lookback: int, horizon: int, width: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(lookback, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
        )
        self.backcast = nn.Linear(width, lookback)
        self.forecast = nn.Linear(width, horizon)

    def forward(self, residual):
        hidden = self.layers(residual)
        return self.backcast(hidden), self.forecast(hidden)


class NBeats(nn.Module):
    def __init__(self, lookback: int, horizon: int, width: int = 32, blocks: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList(NBeatsBlock(lookback, horizon, width) for _ in range(blocks))
        self.horizon = horizon

    def forward(self, history: torch.Tensor):
        residual = history
        forecast = history.new_zeros(history.shape[0], self.horizon)
        for block in self.blocks:
            backcast, addition = block(residual)
            residual = residual - backcast
            forecast = forecast + addition
        return forecast, residual


if __name__ == "__main__":
    model = NBeats(12, 4)
    prediction, remaining = model(torch.randn(2, 12))
    prediction.sum().backward()
    print("forecast:", tuple(prediction.shape), "residual:", tuple(remaining.shape))
