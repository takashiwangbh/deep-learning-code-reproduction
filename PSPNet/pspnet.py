"""PSPNet with pyramid pooling at 1, 2, 3 and 6 spatial bins."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PyramidPooling(nn.Module):
    def __init__(self, in_channels: int, branch_channels: int, bins=(1, 2, 3, 6)):
        super().__init__()
        self.bins = bins
        self.branches = nn.ModuleList(
            nn.Conv2d(in_channels, branch_channels, 1) for _ in bins)
        self.fuse = nn.Sequential(nn.Conv2d(in_channels + branch_channels * len(bins),
                                            in_channels, 3, padding=1, bias=False),
                                  nn.GroupNorm(math.gcd(4, in_channels), in_channels), nn.ReLU())

    def forward(self, feature: torch.Tensor):
        size = feature.shape[-2:]
        pooled = [feature]
        for bins, projection in zip(self.bins, self.branches):
            context = F.adaptive_avg_pool2d(feature, (bins, bins))
            pooled.append(F.interpolate(F.relu(projection(context)), size=size,
                                        mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(pooled, dim=1))


class PSPNet(nn.Module):
    def __init__(self, classes: int = 4, width: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.context = PyramidPooling(width, width // 4)
        self.classifier = nn.Conv2d(width, classes, 1)

    def forward(self, image: torch.Tensor):
        spatial = image.shape[-2:]
        score = self.classifier(self.context(self.encoder(image)))
        return F.interpolate(score, size=spatial, mode="bilinear", align_corners=False)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = PSPNet(classes=3, width=16)
    logits = model(torch.randn(2, 3, 48, 40))
    logits.mean().backward()
    print("PSPNet segmentation logits:", tuple(logits.shape))
