"""Barlow Twins: make the cross-correlation of paired views approach identity."""

import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, width: int = 32, projection: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.projector = nn.Sequential(nn.Linear(width, projection), nn.ReLU(),
                                       nn.Linear(projection, projection))

    def forward(self, images: torch.Tensor):
        return self.projector(self.backbone(images))


def cross_correlation(z_a: torch.Tensor, z_b: torch.Tensor):
    if z_a.ndim != 2 or z_a.shape != z_b.shape or z_a.shape[0] < 2:
        raise ValueError("Expected matching [batch,features] tensors with batch >= 2")
    a = (z_a - z_a.mean(0)) / torch.sqrt(z_a.var(0, unbiased=False) + 1e-5)
    b = (z_b - z_b.mean(0)) / torch.sqrt(z_b.var(0, unbiased=False) + 1e-5)
    return a.T @ b / z_a.shape[0]


def barlow_loss(z_a: torch.Tensor, z_b: torch.Tensor, redundancy_weight: float = 0.005):
    correlation = cross_correlation(z_a, z_b)
    diagonal = (correlation.diagonal() - 1).square().sum()
    off_diagonal = correlation - torch.diag(correlation.diagonal())
    return diagonal + redundancy_weight * off_diagonal.square().sum()


if __name__ == "__main__":
    torch.manual_seed(0)
    model = Encoder(projection=16)
    images = torch.randn(4, 3, 16, 16)
    loss = barlow_loss(model(images), model(images + 0.1 * torch.randn_like(images)))
    loss.backward()
    print("Barlow Twins loss:", round(loss.item(), 4))
