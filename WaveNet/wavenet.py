"""Autoregressive WaveNet blocks with causal, dilated and gated convolutions."""

import torch
from torch import nn
from torch.nn import functional as F


class GatedCausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.filter_gate = nn.Conv1d(channels, 2 * channels, 2, dilation=dilation)
        self.residual_skip = nn.Conv1d(channels, 2 * channels, 1)

    def forward(self, x):
        filters, gates = self.filter_gate(F.pad(x, (self.dilation, 0))).chunk(2, dim=1)
        gated = torch.tanh(filters) * torch.sigmoid(gates)
        residual, skip = self.residual_skip(gated).chunk(2, dim=1)
        return x + residual, skip


class WaveNet(nn.Module):
    def __init__(self, vocabulary_size: int = 32, channels: int = 16, layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, channels)
        self.blocks = nn.ModuleList(GatedCausalBlock(channels, 2 ** index) for index in range(layers))
        self.head = nn.Sequential(nn.ReLU(), nn.Conv1d(channels, channels, 1), nn.ReLU(),
                                  nn.Conv1d(channels, vocabulary_size, 1))

    def forward(self, tokens):
        """Output at t predicts t+1; no convolution can read future token positions."""
        x = self.embedding(tokens).transpose(1, 2)
        skips = []
        for block in self.blocks:
            x, skip = block(x)
            skips.append(skip)
        return self.head(sum(skips)).transpose(1, 2)


if __name__ == "__main__":
    model = WaveNet()
    logits = model(torch.randint(0, 32, (2, 10)))
    logits.sum().backward()
    print("WaveNet logits:", tuple(logits.shape))
