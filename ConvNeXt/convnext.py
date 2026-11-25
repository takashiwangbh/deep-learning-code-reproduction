"""ConvNeXt block: large depthwise kernel, channels-last normalization and MLP."""

import torch
from torch import nn


class ConvNeXtBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.depthwise = nn.Conv2d(width, width, 7, padding=3, groups=width)
        self.norm = nn.LayerNorm(width)
        self.pointwise_1 = nn.Linear(width, 4 * width)
        self.pointwise_2 = nn.Linear(4 * width, width)
        self.gamma = nn.Parameter(torch.ones(width) * 1e-6)

    def forward(self, x):
        branch = self.depthwise(x).permute(0, 2, 3, 1)
        branch = self.pointwise_2(torch.nn.functional.gelu(self.pointwise_1(self.norm(branch))))
        return x + (self.gamma * branch).permute(0, 3, 1, 2)


class ConvNeXt(nn.Module):
    def __init__(self, classes: int = 10, width: int = 16):
        super().__init__()
        self.stem = nn.Conv2d(3, width, 4, stride=4)
        self.blocks = nn.Sequential(ConvNeXtBlock(width), ConvNeXtBlock(width))
        self.downsample = nn.Conv2d(width, width * 2, 2, stride=2)
        self.last = ConvNeXtBlock(width * 2)
        self.norm = nn.LayerNorm(width * 2)
        self.head = nn.Linear(width * 2, classes)

    def forward(self, x):
        z = self.last(self.downsample(self.blocks(self.stem(x)))).mean(dim=(-2, -1))
        return self.head(self.norm(z))


if __name__ == "__main__":
    y = ConvNeXt()(torch.randn(2, 3, 32, 32))
    y.sum().backward()
    print("ConvNeXt logits:", tuple(y.shape))
