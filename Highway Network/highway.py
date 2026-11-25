"""Highway layers mix a learned transform with an input-carry path."""

import torch
from torch import nn
from torch.nn import functional as F


class HighwayLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.transform = nn.Linear(width, width)
        self.gate = nn.Linear(width, width)
        nn.init.constant_(self.gate.bias, -2.0)  # initially favor carry path

    def forward(self, x: torch.Tensor):
        transformed = F.relu(self.transform(x))
        gate = torch.sigmoid(self.gate(x))
        return gate * transformed + (1 - gate) * x


class HighwayNetwork(nn.Module):
    def __init__(self, features: int, width: int, classes: int, depth: int = 8):
        super().__init__()
        self.input = nn.Linear(features, width)
        self.layers = nn.ModuleList(HighwayLayer(width) for _ in range(depth))
        self.classifier = nn.Linear(width, classes)

    def forward(self, x: torch.Tensor):
        hidden = F.relu(self.input(x))
        for layer in self.layers:
            hidden = layer(hidden)
        return self.classifier(hidden)


if __name__ == "__main__":
    torch.manual_seed(0)
    net = HighwayNetwork(10, 16, 3, depth=8)
    logits = net(torch.randn(4, 10))
    logits.sum().backward()
    print("Highway logits:", tuple(logits.shape))
