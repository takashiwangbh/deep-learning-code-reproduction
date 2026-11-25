"""Depthwise separable convolutions with Xception-style residual blocks."""

import torch
from torch import nn


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )


class XceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.path = nn.Sequential(
            nn.ReLU(), SeparableConv(in_channels, out_channels),
            nn.ReLU(), SeparableConv(out_channels, out_channels),
            nn.MaxPool2d(3, stride=stride, padding=1) if stride > 1 else nn.Identity(),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride) if in_channels != out_channels or stride > 1 else nn.Identity()

    def forward(self, x):
        return self.path(x) + self.shortcut(x)


class Xception(nn.Module):
    def __init__(self, classes: int = 10):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.blocks = nn.Sequential(XceptionBlock(16, 32, 2), XceptionBlock(32, 64, 2), XceptionBlock(64, 64))
        self.head = nn.Linear(64, classes)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)).mean(dim=(-2, -1)))


if __name__ == "__main__":
    y = Xception()(torch.randn(2, 3, 32, 32))
    y.sum().backward()
    print("Xception logits:", tuple(y.shape))
