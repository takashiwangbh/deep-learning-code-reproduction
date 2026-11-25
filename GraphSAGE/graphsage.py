"""GraphSAGE mean aggregation for node features, without per-node embeddings."""

import torch
from torch import nn
from torch.nn import functional as F


class MeanSAGELayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(2 * in_features, out_features)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """x:[N,F], adjacency:[N,N] with neighbors but no self loops."""
        neighbors = adjacency.to(dtype=x.dtype)
        degree = neighbors.sum(dim=-1, keepdim=True).clamp_min(1)
        neighbor_mean = neighbors @ x / degree
        hidden = F.relu(self.linear(torch.cat((x, neighbor_mean), dim=-1)))
        return F.normalize(hidden, p=2, dim=-1)


class GraphSAGE(nn.Module):
    def __init__(self, in_features: int, hidden: int, classes: int):
        super().__init__()
        self.first = MeanSAGELayer(in_features, hidden)
        self.second = MeanSAGELayer(hidden, hidden)
        self.classifier = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor):
        x = self.first(x, adjacency)
        x = self.second(x, adjacency)
        return self.classifier(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    edges = torch.tensor([[0, 1, 0, 0], [1, 0, 1, 0],
                          [0, 1, 0, 1], [0, 0, 1, 0]])
    logits = GraphSAGE(5, 8, 3)(torch.randn(4, 5), edges)
    logits.sum().backward()
    print("GraphSAGE node logits:", tuple(logits.shape))
