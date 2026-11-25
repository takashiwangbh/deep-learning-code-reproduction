"""ShuffleNet v1 group pointwise convolutions and channel shuffle."""

import torch
from torch import nn


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch, channels, height, width = x.shape
    if channels % groups:
        raise ValueError("channels must divide into groups")
    return x.reshape(batch, groups, channels // groups, height, width).transpose(1, 2).reshape(batch, channels, height, width)


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 2, stride: int = 1):
        super().__init__()
        if stride not in (1, 2) or (stride == 1 and in_channels != out_channels):
            raise ValueError("stride-1 unit must preserve channels")
        branch_channels = out_channels if stride == 1 else out_channels - in_channels
        middle = max(groups, branch_channels // 4)
        middle = ((middle + groups - 1) // groups) * groups
        if in_channels % groups or branch_channels % groups:
            raise ValueError("grouped pointwise convolution needs divisible channel counts")
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, middle, 1, groups=groups, bias=False),
            nn.BatchNorm2d(middle), nn.ReLU(),
        )
        self.groups = groups
        self.depthwise = nn.Sequential(
            nn.Conv2d(middle, middle, 3, stride=stride, padding=1, groups=middle, bias=False),
            nn.BatchNorm2d(middle),
        )
        self.last = nn.Sequential(
            nn.Conv2d(middle, branch_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(branch_channels),
        )
        self.stride = stride

    def forward(self, x):
        branch = self.last(self.depthwise(channel_shuffle(self.first(x), self.groups)))
        shortcut = x if self.stride == 1 else nn.functional.avg_pool2d(x, 3, stride=2, padding=1)
        return torch.relu(shortcut + branch) if self.stride == 1 else torch.cat([shortcut, branch], dim=1)


class ShuffleNet(nn.Module):
    def __init__(self, classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 24, 3, stride=2, padding=1), nn.ReLU())
        self.blocks = nn.Sequential(ShuffleUnit(24, 48, stride=2), ShuffleUnit(48, 48))
        self.head = nn.Linear(48, classes)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)).mean(dim=(-2, -1)))


if __name__ == "__main__":
    y = ShuffleNet()(torch.randn(2, 3, 32, 32))
    y.sum().backward()
    print("ShuffleNet logits:", tuple(y.shape))
