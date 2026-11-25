"""DeepFM shares field embeddings between a factorization machine and an MLP."""

import torch
from torch import nn


class DeepFM(nn.Module):
    def __init__(self, field_sizes: list[int], embedding_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.field_sizes = field_sizes
        offsets = [0]
        for size in field_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        self.register_buffer("offsets", torch.tensor(offsets))
        total_features = sum(field_sizes)
        self.first_order = nn.Embedding(total_features, 1)
        self.embedding = nn.Embedding(total_features, embedding_dim)
        self.deep = nn.Sequential(nn.Linear(len(field_sizes) * embedding_dim, hidden),
                                  nn.ReLU(), nn.Linear(hidden, 1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        if fields.shape[-1] != len(self.field_sizes):
            raise ValueError("Expected one categorical ID for each field")
        ids = fields + self.offsets
        v = self.embedding(ids)  # [B,fields,embedding_dim]
        linear = self.first_order(ids).sum(dim=1)
        fm = 0.5 * ((v.sum(dim=1) ** 2 - (v ** 2).sum(dim=1)).sum(dim=-1, keepdim=True))
        deep = self.deep(v.flatten(start_dim=1))
        return (linear + fm + deep + self.bias).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = DeepFM([5, 8, 6])
    logits = model(torch.tensor([[0, 1, 2], [3, 2, 1], [4, 7, 5]]))
    nn.BCEWithLogitsLoss()(logits, torch.tensor([1., 0., 1.])).backward()
    print("DeepFM logits:", tuple(logits.shape))
