"""Causal dilated residual convolutions for temporal convolutional networks."""

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(CausalConv1d(channels, kernel_size, dilation), nn.ReLU(),
                                 CausalConv1d(channels, kernel_size, dilation), nn.ReLU())

    def forward(self, x):
        return x + self.net(x)


class TCN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, levels: int = 3):
        super().__init__()
        self.input = nn.Conv1d(input_size, hidden_size, 1)
        self.blocks = nn.Sequential(*(TemporalBlock(hidden_size, 3, 2 ** index) for index in range(levels)))
        self.output = nn.Conv1d(hidden_size, output_size, 1)

    def forward(self, x: torch.Tensor):
        return self.output(self.blocks(self.input(x.transpose(1, 2)))).transpose(1, 2)


if __name__ == "__main__":
    model = TCN(4, 8, 3)
    y = model(torch.randn(2, 9, 4))
    y.sum().backward()
    print("TCN output:", tuple(y.shape))
