"""两层图卷积网络（GCN），手写邻接矩阵归一化与消息聚合。

Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks:
https://arxiv.org/abs/1609.02907
"""

import torch
from torch import nn
from torch.nn import functional as F


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """A_hat = D^-1/2 (A + I) D^-1/2，孤立节点靠自环保持自身特征。"""
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square [N, N] matrix")
    adjacency = adjacency + torch.eye(adjacency.size(0), device=adjacency.device, dtype=adjacency.dtype)
    degree_inv_sqrt = adjacency.sum(dim=1).clamp_min(1).pow(-0.5)
    return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]


class GraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, features: torch.Tensor, normalized_adjacency: torch.Tensor) -> torch.Tensor:
        # 每个节点先线性变换，再汇聚一阶邻居（含自身）。
        return normalized_adjacency @ self.linear(features)


class GCN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, classes: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GraphConv(in_features, hidden_features)
        self.conv2 = GraphConv(hidden_features, classes)
        self.dropout = dropout

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        normalized = normalize_adjacency(adjacency)
        hidden = F.relu(self.conv1(features, normalized))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.conv2(hidden, normalized)  # [节点数, 类别数] logits


if __name__ == "__main__":
    model = GCN(in_features=5, hidden_features=8, classes=3)
    features = torch.randn(4, 5)
    adjacency = torch.tensor([[0., 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    logits = model(features, adjacency)
    F.cross_entropy(logits[:2], torch.tensor([0, 2])).backward()  # 只有前两个有标签
    print(f"node_logits={tuple(logits.shape)}")
