"""Graph Attention Network: masked, learned attention over each node's neighbors."""

import torch
from torch import nn
from torch.nn import functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, heads: int = 4,
                 concat: bool = True):
        super().__init__()
        self.heads, self.out_features, self.concat = heads, out_features, concat
        self.weight = nn.Parameter(torch.empty(heads, in_features, out_features))
        self.attn_source = nn.Parameter(torch.empty(heads, out_features))
        self.attn_target = nn.Parameter(torch.empty(heads, out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attn_source)
        nn.init.xavier_uniform_(self.attn_target)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor):
        """x:[N,F]; adjacency:[N,N], self loops added here."""
        n = x.shape[0]
        projected = torch.einsum("nf,hfo->hno", x, self.weight)
        source = (projected * self.attn_source[:, None, :]).sum(dim=-1)
        target = (projected * self.attn_target[:, None, :]).sum(dim=-1)
        scores = F.leaky_relu(source[:, :, None] + target[:, None, :], 0.2)
        mask = adjacency.bool() | torch.eye(n, device=x.device, dtype=torch.bool)
        scores = scores.masked_fill(~mask[None, :, :], torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        values = torch.matmul(weights, projected).transpose(0, 1)  # [N,H,D]
        return values.reshape(n, -1) if self.concat else values.mean(dim=1)


class GAT(nn.Module):
    def __init__(self, in_features: int, hidden: int, classes: int, heads: int = 4):
        super().__init__()
        self.first = GraphAttentionLayer(in_features, hidden, heads)
        self.second = GraphAttentionLayer(hidden * heads, classes, 1, concat=False)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor):
        return self.second(F.elu(self.first(x, adjacency)), adjacency)


if __name__ == "__main__":
    torch.manual_seed(0)
    adjacency = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    model = GAT(5, 4, 2)
    logits = model(torch.randn(3, 5), adjacency)
    logits.sum().backward()
    print("GAT node logits:", tuple(logits.shape))
