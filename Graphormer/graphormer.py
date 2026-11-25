"""Graphormer: degree encodings plus shortest-path and edge attention biases."""

from collections import deque
import math

import torch
from torch import nn


def shortest_path_features(adjacency: torch.Tensor, edge_types: torch.Tensor):
    """Dense teaching BFS; edge ID 0 means no edge. Return distances and path IDs."""
    batch, nodes, _ = adjacency.shape
    if edge_types.shape != adjacency.shape:
        raise ValueError("edge_types must match adjacency [B,N,N]")
    distance = torch.full((batch, nodes, nodes), nodes + 1, dtype=torch.long,
                          device=adjacency.device)
    paths = torch.zeros(batch, nodes, nodes, max(nodes - 1, 1), dtype=torch.long,
                        device=adjacency.device)
    for b in range(batch):
        for source in range(nodes):
            distance[b, source, source] = 0
            queue = deque([source])
            route = {source: []}
            while queue:
                node = queue.popleft()
                for neighbor in torch.where(adjacency[b, node].bool())[0].tolist():
                    if neighbor in route:
                        continue
                    edge_id = int(edge_types[b, node, neighbor])
                    if edge_id == 0:
                        raise ValueError("Present edges need a positive edge-type ID")
                    route[neighbor] = route[node] + [edge_id]
                    distance[b, source, neighbor] = len(route[neighbor])
                    paths[b, source, neighbor, :len(route[neighbor])] = torch.tensor(
                        route[neighbor], device=adjacency.device)
                    queue.append(neighbor)
    return distance, paths


class Graphormer(nn.Module):
    def __init__(self, input_dim: int, classes: int, width: int = 32,
                 heads: int = 4, max_degree: int = 16, max_distance: int = 16,
                 edge_vocab: int = 8):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads, self.head_dim, self.max_distance = heads, width // heads, max_distance
        self.node_projection = nn.Linear(input_dim, width)
        self.degree = nn.Embedding(max_degree + 1, width)
        self.graph_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.spatial_bias = nn.Embedding(max_distance + 2, heads)
        self.edge_bias = nn.Embedding(edge_vocab, heads, padding_idx=0)
        self.query, self.key, self.value = (nn.Linear(width, width) for _ in range(3))
        self.output = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                         nn.Linear(2 * width, width))
        self.final_norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, classes)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor,
                edge_types: torch.Tensor):
        """nodes [B,N,F], adjacency/edge_types [B,N,N] -> graph logits [B,C]."""
        batch, count, _ = nodes.shape
        if adjacency.shape != (batch, count, count):
            raise ValueError("adjacency must be [B,N,N]")
        degrees = adjacency.long().sum(-1).clamp_max(self.degree.num_embeddings - 1)
        encoded = self.node_projection(nodes) + self.degree(degrees)
        encoded = torch.cat((self.graph_token.expand(batch, -1, -1), encoded), dim=1)

        distance, path_edges = shortest_path_features(adjacency, edge_types)
        spatial = self.spatial_bias(distance.clamp_max(self.max_distance + 1))
        counts = path_edges.ne(0).sum(dim=-1, keepdim=True).clamp_min(1)
        edge = self.edge_bias(path_edges).sum(dim=-2) / counts
        # The graph token attends to all nodes with zero structural bias.
        bias = encoded.new_zeros(batch, self.heads, count + 1, count + 1)
        bias[:, :, 1:, 1:] = (spatial + edge).permute(0, 3, 1, 2)

        def split(projection):
            return projection(encoded).reshape(batch, count + 1, self.heads,
                                               self.head_dim).transpose(1, 2)
        q, k, v = split(self.query), split(self.key), split(self.value)
        weights = ((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) + bias).softmax(-1)
        attended = (weights @ v).transpose(1, 2).reshape(batch, count + 1, -1)
        encoded = self.norm(encoded + self.output(attended))
        encoded = self.final_norm(encoded + self.feedforward(encoded))
        return self.classifier(encoded[:, 0])


if __name__ == "__main__":
    torch.manual_seed(0)
    model = Graphormer(5, 3)
    adjacency = torch.tensor([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]])
    edge_types = adjacency.clone()
    logits = model(torch.randn(1, 3, 5), adjacency, edge_types)
    logits.sum().backward()
    print("Graphormer graph logits:", tuple(logits.shape))
