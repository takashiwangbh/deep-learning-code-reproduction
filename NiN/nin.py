"""Network in Network: local MLPs made from 1x1 convolutions."""

import torch
from torch import nn


class MLPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.ReLU(inplace=True),
        )


class NiN(nn.Module):
    def __init__(self, classes: int = 10, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            MLPConv(3, width), nn.MaxPool2d(2),
            MLPConv(width, width * 2), nn.MaxPool2d(2),
            MLPConv(width * 2, width * 2),
        )
        self.class_map = nn.Conv2d(width * 2, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.class_map(self.features(x)).mean(dim=(-2, -1))


if __name__ == "__main__":
    model = NiN(width=8)
    y = model(torch.randn(2, 3, 32, 32))
    y.sum().backward()
    print("NiN logits:", tuple(y.shape))
