"""ResNeXt bottlenecks with grouped 3x3 convolutions (cardinality)."""

import torch
from torch import nn


class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cardinality: int = 4,
                 group_width: int = 4, stride: int = 1):
        super().__init__()
        middle = cardinality * group_width
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, middle, 1, bias=False), nn.BatchNorm2d(middle), nn.ReLU(),
            nn.Conv2d(middle, middle, 3, stride=stride, padding=1, groups=cardinality, bias=False),
            nn.BatchNorm2d(middle), nn.ReLU(),
            nn.Conv2d(middle, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity() if stride == 1 and in_channels == out_channels else
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                          nn.BatchNorm2d(out_channels))
        )

    def forward(self, x):
        return torch.relu(self.transform(x) + self.shortcut(x))


class ResNeXt(nn.Module):
    def __init__(self, classes: int = 10, cardinality: int = 4, group_width: int = 4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.blocks = nn.Sequential(
            ResNeXtBlock(32, 32, cardinality, group_width),
            ResNeXtBlock(32, 64, cardinality, group_width, stride=2),
            ResNeXtBlock(64, 64, cardinality, group_width),
        )
        self.head = nn.Linear(64, classes)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)).mean(dim=(-2, -1)))


if __name__ == "__main__":
    y = ResNeXt()(torch.randn(2, 3, 32, 32))
    y.sum().backward()
    print("ResNeXt logits:", tuple(y.shape))
