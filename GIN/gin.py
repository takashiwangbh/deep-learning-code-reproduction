"""Graph Isomorphism Network with sum aggregation and learnable epsilon."""

import torch
from torch import nn


class GINLayer(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.epsilon = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(nn.Linear(features, features), nn.ReLU(),
                                 nn.Linear(features, features))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor):
        """adjacency excludes diagonal; sum preserves neighbor multiplicity."""
        neighbors = adjacency.to(x.dtype) @ x
        return self.mlp((1 + self.epsilon) * x + neighbors)


class GINClassifier(nn.Module):
    def __init__(self, features: int, classes: int, layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(GINLayer(features) for _ in range(layers))
        self.readout = nn.Linear(features * (layers + 1), classes)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor):
        """A single graph: x [N,F], output graph logits [classes]."""
        pooled = [x.sum(dim=0)]
        for layer in self.layers:
            x = layer(x, adjacency)
            pooled.append(x.sum(dim=0))
        return self.readout(torch.cat(pooled))


if __name__ == "__main__":
    torch.manual_seed(0)
    adjacency = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    logits = GINClassifier(5, 2)(torch.randn(3, 5), adjacency)
    logits.sum().backward()
    print("GIN graph logits:", tuple(logits.shape))
