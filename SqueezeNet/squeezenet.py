"""SqueezeNet Fire modules: squeeze 1x1, then expand 1x1 and 3x3."""

import torch
from torch import nn


class Fire(nn.Module):
    def __init__(self, in_channels: int, squeeze: int, expand: int):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(in_channels, squeeze, 1), nn.ReLU(inplace=True))
        self.expand_1 = nn.Conv2d(squeeze, expand, 1)
        self.expand_3 = nn.Conv2d(squeeze, expand, 3, padding=1)

    def forward(self, x):
        squeezed = self.squeeze(x)
        return torch.cat([torch.relu(self.expand_1(squeezed)), torch.relu(self.expand_3(squeezed))], dim=1)


class SqueezeNet(nn.Module):
    def __init__(self, classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            Fire(16, 4, 16), Fire(32, 4, 16), nn.MaxPool2d(2),
            Fire(32, 8, 32), Fire(64, 8, 32),
        )
        self.class_map = nn.Conv2d(64, classes, 1)

    def forward(self, x):
        return self.class_map(self.features(x)).mean(dim=(-2, -1))


if __name__ == "__main__":
    y = SqueezeNet()(torch.randn(2, 3, 32, 32))
    y.sum().backward()
    print("SqueezeNet logits:", tuple(y.shape))
