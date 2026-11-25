"""A generic residual block with an explicit projection shortcut."""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity() if in_channels == out_channels and stride == 1 else
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                          nn.BatchNorm2d(out_channels))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.transform(x) + self.shortcut(x))


if __name__ == "__main__":
    block = ResidualBlock(8, 16, stride=2)
    y = block(torch.randn(2, 8, 16, 16))
    y.sum().backward()
    print("residual output:", tuple(y.shape))
