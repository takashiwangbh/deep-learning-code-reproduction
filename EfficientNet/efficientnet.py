"""Compact EfficientNet: MBConv, SE, SiLU, and compound-scaled stages."""

import math
import torch
from torch import nn


def scaled_channels(channels: int, width_scale: float, divisor: int = 8) -> int:
    value = channels * width_scale
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, squeeze_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, squeeze_channels, 1), nn.SiLU(),
            nn.Conv2d(squeeze_channels, channels, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class MBConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expansion: int = 4, stride: int = 1):
        super().__init__()
        middle = in_channels * expansion
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, middle, 1, bias=False), nn.BatchNorm2d(middle), nn.SiLU(),
            nn.Conv2d(middle, middle, 3, stride=stride, padding=1, groups=middle, bias=False),
            nn.BatchNorm2d(middle), nn.SiLU(),
            SqueezeExcitation(middle, max(1, in_channels // 4)),
            nn.Conv2d(middle, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels),
        )
        self.use_skip = stride == 1 and in_channels == out_channels

    def forward(self, x):
        result = self.layers(x)
        return x + result if self.use_skip else result


class EfficientNet(nn.Module):
    def __init__(self, classes: int = 10, width_scale: float = 1.0, depth_scale: float = 1.0):
        super().__init__()
        if width_scale <= 0 or depth_scale <= 0:
            raise ValueError("scales must be positive")
        stem_width = scaled_channels(16, width_scale)
        self.stem = nn.Sequential(nn.Conv2d(3, stem_width, 3, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(stem_width), nn.SiLU())
        stages = []
        current = stem_width
        for channels, repeats, stride in [(16, 1, 1), (24, 2, 2), (40, 2, 2)]:
            target = scaled_channels(channels, width_scale)
            for index in range(math.ceil(repeats * depth_scale)):
                stages.append(MBConv(current, target, stride=stride if index == 0 else 1))
                current = target
        self.stages = nn.Sequential(*stages)
        self.head = nn.Linear(current, classes)

    def forward(self, x):
        return self.head(self.stages(self.stem(x)).mean(dim=(-2, -1)))


if __name__ == "__main__":
    y = EfficientNet()(torch.randn(2, 3, 32, 32))
    y.sum().backward()
    print("EfficientNet logits:", tuple(y.shape))
