"""Squeeze-and-Excitation gating inside a residual CNN block."""

import torch
from torch import nn


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels),
            SqueezeExcitation(channels),
        )

    def forward(self, x):
        return torch.relu(x + self.residual(x))


class SENet(nn.Module):
    def __init__(self, classes: int = 10, width: int = 16):
        super().__init__()
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.blocks = nn.Sequential(SEBlock(width), SEBlock(width))
        self.head = nn.Linear(width, classes)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)).mean(dim=(-2, -1)))


if __name__ == "__main__":
    y = SENet()(torch.randn(2, 3, 32, 32))
    y.sum().backward()
    print("SENet logits:", tuple(y.shape))
