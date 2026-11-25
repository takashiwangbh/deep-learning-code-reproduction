"""VICReg: invariance, per-dimension variance and off-diagonal covariance."""

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


def variance_penalty(z: torch.Tensor):
    standard_deviation = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(1 - standard_deviation).mean()


def covariance_penalty(z: torch.Tensor):
    centered = z - z.mean(dim=0)
    covariance = centered.T @ centered / (z.shape[0] - 1)
    off_diagonal = covariance - torch.diag(covariance.diagonal())
    return off_diagonal.square().sum() / z.shape[1]


def vicreg_loss(z_a: torch.Tensor, z_b: torch.Tensor,
                invariance_weight: float = 25.0, variance_weight: float = 25.0,
                covariance_weight: float = 1.0):
    if z_a.ndim != 2 or z_a.shape != z_b.shape or z_a.shape[0] < 2:
        raise ValueError("Expected matching [batch,features] tensors with batch >= 2")
    invariance = F.mse_loss(z_a, z_b)
    variance = 0.5 * (variance_penalty(z_a) + variance_penalty(z_b))
    covariance = covariance_penalty(z_a) + covariance_penalty(z_b)
    return (invariance_weight * invariance + variance_weight * variance
            + covariance_weight * covariance)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = Encoder(projection=16)
    images = torch.randn(4, 3, 16, 16)
    loss = vicreg_loss(model(images), model(images + 0.1 * torch.randn_like(images)))
    loss.backward()
    print("VICReg loss:", round(loss.item(), 4))
