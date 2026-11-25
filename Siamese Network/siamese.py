"""Shared-weight twin encoder and contrastive loss."""

import torch
from torch import nn
from torch.nn import functional as F


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_size: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, embedding_size),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        return self.encoder(left), self.encoder(right)


def contrastive_loss(left: torch.Tensor, right: torch.Tensor, same: torch.Tensor, margin: float = 1.0):
    """same=1 means matching pair, same=0 means nonmatching pair."""
    distance = torch.norm(left - right, dim=-1)
    same = same.to(distance.dtype)
    return (same * distance.square() + (1 - same) * F.relu(margin - distance).square()).mean()


if __name__ == "__main__":
    model = SiameseNetwork()
    a, b = model(torch.randn(2, 1, 28, 28), torch.randn(2, 1, 28, 28))
    loss = contrastive_loss(a, b, torch.tensor([1., 0.]))
    loss.backward()
    print("embeddings:", tuple(a.shape), tuple(b.shape), "loss:", float(loss))
