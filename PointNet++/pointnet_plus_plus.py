"""PointNet++ single-scale set abstraction: FPS, radius grouping and local PointNet."""

import torch
from torch import nn


@torch.no_grad()
def farthest_point_sample(xyz: torch.Tensor, count: int):
    """xyz [B,N,3] -> [B,count] distinct indices, seeded by geometric spread."""
    batch, points, _ = xyz.shape
    if not 0 < count <= points:
        raise ValueError("count must be between 1 and number of points")
    chosen = torch.zeros(batch, count, dtype=torch.long, device=xyz.device)
    distance = torch.full((batch, points), float("inf"), device=xyz.device)
    centroid = xyz.mean(dim=1, keepdim=True)
    current = ((xyz - centroid) ** 2).sum(-1).argmax(-1)
    rows = torch.arange(batch, device=xyz.device)
    for step in range(count):
        chosen[:, step] = current
        center = xyz[rows, current][:, None, :]
        distance = torch.minimum(distance, ((xyz - center) ** 2).sum(-1))
        distance[rows[:, None], chosen[:, :step + 1]] = -1
        current = distance.argmax(-1)
    return chosen


@torch.no_grad()
def radius_neighbors(xyz: torch.Tensor, centers: torch.Tensor,
                     radius: float, samples: int):
    """[B,S,K] indices; unavailable slots repeat the nearest point."""
    if radius <= 0 or samples < 1:
        raise ValueError("radius and samples must be positive")
    distance = torch.cdist(centers, xyz)
    nearest = distance.argmin(-1, keepdim=True)
    valid = distance <= radius
    candidate = distance.masked_fill(~valid, float("inf"))
    count = min(samples, xyz.shape[1])
    picked = candidate.topk(count, largest=False).indices
    picked_distance = candidate.gather(-1, picked)
    picked = torch.where(torch.isfinite(picked_distance), picked, nearest.expand_as(picked))
    if count < samples:
        picked = torch.cat((picked, nearest.expand(*nearest.shape[:-1], samples - count)), -1)
    return picked


class SetAbstraction(nn.Module):
    def __init__(self, input_features: int, output_features: int, centers: int,
                 neighbors: int, radius: float):
        super().__init__()
        self.centers, self.neighbors, self.radius = centers, neighbors, radius
        self.local_mlp = nn.Sequential(
            nn.Conv2d(3 + input_features, output_features, 1), nn.ReLU(),
            nn.Conv2d(output_features, output_features, 1), nn.ReLU(),
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor | None = None):
        batch = torch.arange(xyz.shape[0], device=xyz.device)[:, None, None]
        center_ids = farthest_point_sample(xyz, self.centers)
        center_xyz = xyz[torch.arange(xyz.shape[0], device=xyz.device)[:, None], center_ids]
        neighbor_ids = radius_neighbors(xyz, center_xyz, self.radius, self.neighbors)
        relative = xyz[batch, neighbor_ids] - center_xyz[:, :, None, :]
        if features is not None:
            relative = torch.cat((relative, features[batch, neighbor_ids]), -1)
        hidden = self.local_mlp(relative.permute(0, 3, 1, 2)).max(dim=-1).values
        return center_xyz, hidden.transpose(1, 2)


class PointNetPlusPlus(nn.Module):
    def __init__(self, classes: int = 10):
        super().__init__()
        self.first = SetAbstraction(0, 32, centers=16, neighbors=8, radius=0.4)
        self.second = SetAbstraction(32, 64, centers=4, neighbors=8, radius=0.8)
        self.classifier = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, classes))

    def forward(self, xyz: torch.Tensor):
        """xyz [B,N,3] with N>=16 -> class logits [B,C]."""
        xyz, features = self.first(xyz)
        _, features = self.second(xyz, features)
        return self.classifier(features.max(dim=1).values)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = PointNetPlusPlus(classes=5)
    logits = model(torch.rand(2, 32, 3))
    logits.sum().backward()
    print("PointNet++ logits:", tuple(logits.shape))
