"""PointNet classification: input transform, shared point MLP and symmetric max."""

import torch
from torch import nn


class InputTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_features = nn.Sequential(nn.Linear(3, 64), nn.ReLU(),
                                            nn.Linear(64, 128), nn.ReLU(),
                                            nn.Linear(128, 256), nn.ReLU())
        self.global_features = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                             nn.Linear(128, 9))
        nn.init.zeros_(self.global_features[-1].weight)
        nn.init.zeros_(self.global_features[-1].bias)

    def forward(self, points: torch.Tensor):
        features = self.point_features(points).max(dim=1).values
        delta = self.global_features(features).view(-1, 3, 3)
        eye = torch.eye(3, device=points.device, dtype=points.dtype)
        return eye + delta


class PointNet(nn.Module):
    def __init__(self, classes: int = 10):
        super().__init__()
        self.transform = InputTransform()
        self.shared_mlp = nn.Sequential(nn.Linear(3, 64), nn.ReLU(),
                                        nn.Linear(64, 128), nn.ReLU(),
                                        nn.Linear(128, 256), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                        nn.Linear(128, classes))

    def forward(self, points: torch.Tensor):
        """points [B,N,3] -> logits [B,classes], invariant to point order."""
        transform = self.transform(points)
        aligned = torch.bmm(points, transform)
        features = self.shared_mlp(aligned).max(dim=1).values
        return self.classifier(features)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = PointNet(4)
    logits = model(torch.randn(2, 16, 3))
    logits.sum().backward()
    print("PointNet logits:", tuple(logits.shape))
