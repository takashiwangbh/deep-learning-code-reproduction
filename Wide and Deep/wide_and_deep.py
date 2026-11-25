"""Joint wide linear memorization and deep embedding generalization."""

import torch
from torch import nn


class WideAndDeep(nn.Module):
    def __init__(self, dense_features: int, categories: int, crosses: int,
                 embedding_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.wide_dense = nn.Linear(dense_features, 1, bias=False)
        self.wide_category = nn.Embedding(categories, 1)
        self.wide_cross = nn.Embedding(crosses, 1)
        self.deep_category = nn.Embedding(categories, embedding_dim)
        self.deep = nn.Sequential(nn.Linear(dense_features + embedding_dim, hidden),
                                  nn.ReLU(), nn.Linear(hidden, 1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, dense: torch.Tensor, category_ids: torch.Tensor,
                cross_ids: torch.Tensor) -> torch.Tensor:
        # category_ids/cross_ids may contain several active fields per example.
        wide = (self.wide_dense(dense) + self.wide_category(category_ids).sum(1)
                + self.wide_cross(cross_ids).sum(1) + self.bias)
        deep_embed = self.deep_category(category_ids).sum(1)
        deep = self.deep(torch.cat((dense, deep_embed), dim=-1))
        return (wide + deep).squeeze(-1)  # logits for BCEWithLogitsLoss


if __name__ == "__main__":
    torch.manual_seed(0)
    model = WideAndDeep(3, 20, 12)
    logits = model(torch.randn(4, 3), torch.randint(0, 20, (4, 2)),
                   torch.randint(0, 12, (4, 1)))
    nn.BCEWithLogitsLoss()(logits, torch.randint(0, 2, (4,)).float()).backward()
    print("Wide & Deep logits:", tuple(logits.shape))
