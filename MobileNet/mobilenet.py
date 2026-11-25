"""MobileNet v1 depthwise-separable convolution, separate from MobileNetV2."""

import torch
from torch import nn


class DepthwiseSeparable(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )


class MobileNet(nn.Module):
    def __init__(self, classes: int = 10, width: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(),
            DepthwiseSeparable(width, width * 2),
            DepthwiseSeparable(width * 2, width * 4, stride=2),
            DepthwiseSeparable(width * 4, width * 4),
        )
        self.head = nn.Linear(width * 4, classes)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(-2, -1)))


if __name__ == "__main__":
    y = MobileNet()(torch.randn(2, 3, 32, 32))
    y.sum().backward()
    print("MobileNet v1 logits:", tuple(y.shape))
