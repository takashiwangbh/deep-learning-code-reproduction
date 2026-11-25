"""PatchTST: patch each univariate channel and share one Transformer across channels."""

import torch
from torch import nn


class PatchTST(nn.Module):
    def __init__(self, history: int, horizon: int, patch: int = 4,
                 stride: int = 2, width: int = 32, heads: int = 4):
        super().__init__()
        if not 0 < patch <= history or stride <= 0:
            raise ValueError("Patch length must fit history and stride must be positive")
        self.history = history
        self.patch, self.stride = patch, stride
        self.patches = 1 + (history - patch) // stride
        self.embedding = nn.Linear(patch, width)
        self.position = nn.Parameter(torch.randn(1, self.patches, width) * 0.02)
        block = nn.TransformerEncoderLayer(width, heads, 2 * width,
                                           dropout=0, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=2)
        self.forecast = nn.Linear(self.patches * width, horizon)

    def forward(self, series: torch.Tensor):
        """series [B,history,C] -> forecast [B,horizon,C]."""
        batch, length, channels = series.shape
        if length != self.history:
            raise ValueError("History length differs from configured length")
        patches = series.transpose(1, 2).unfold(-1, self.patch, self.stride)
        tokens = self.embedding(patches.reshape(batch * channels, self.patches, self.patch))
        encoded = self.encoder(tokens + self.position)
        future = self.forecast(encoded.flatten(start_dim=1))
        return future.reshape(batch, channels, -1).transpose(1, 2)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = PatchTST(history=16, horizon=4)
    prediction = model(torch.randn(2, 16, 3))
    prediction.square().mean().backward()
    print("PatchTST forecast:", tuple(prediction.shape))
