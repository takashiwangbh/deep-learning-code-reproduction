"""Dynamic Graph CNN: rebuild k-nearest-neighbor graph in feature space."""

import torch
from torch import nn


def knn_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """x [B,N,C]; exclude each point itself. Neighbor choice is discrete."""
    b, n, _ = x.shape
    if not 0 < k < n:
        raise ValueError("k must be smaller than the number of points")
    with torch.no_grad():
        distances = torch.cdist(x.detach(), x.detach())
        eye = torch.eye(n, device=x.device, dtype=torch.bool)
        distances.masked_fill_(eye[None], float("inf"))
        return distances.topk(k, dim=-1, largest=False).indices


class EdgeConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, k: int = 8):
        super().__init__()
        self.k = k
        self.edge_mlp = nn.Sequential(nn.Conv2d(2 * in_features, out_features, 1),
                                      nn.ReLU(), nn.Conv2d(out_features, out_features, 1),
                                      nn.ReLU())

    def forward(self, x: torch.Tensor):
        b, n, c = x.shape
        neighbors = knn_indices(x, self.k)
        batch = torch.arange(b, device=x.device)[:, None, None]
        x_j = x[batch, neighbors]
        x_i = x[:, :, None, :].expand_as(x_j)
        edge = torch.cat((x_i, x_j - x_i), dim=-1).permute(0, 3, 1, 2)
        return self.edge_mlp(edge).max(dim=-1).values.transpose(1, 2)


class DGCNN(nn.Module):
    def __init__(self, classes: int = 10, k: int = 8):
        super().__init__()
        self.first = EdgeConv(3, 32, k)
        self.second = EdgeConv(32, 64, k)
        self.classifier = nn.Sequential(nn.Linear((32 + 64) * 2, 128), nn.ReLU(),
                                        nn.Linear(128, classes))

    def forward(self, points: torch.Tensor):
        first = self.first(points)
        second = self.second(first)  # new neighbors selected from learned features
        features = torch.cat((first, second), dim=-1)
        global_features = torch.cat((features.max(dim=1).values,
                                     features.mean(dim=1)), dim=-1)
        return self.classifier(global_features)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = DGCNN(4, k=3)
    logits = model(torch.randn(2, 12, 3))
    logits.sum().backward()
    print("DGCNN logits:", tuple(logits.shape))
