"""NeuMF: concatenate generalized matrix factorization and MLP interactions."""

import torch
from torch import nn


class NeuMF(nn.Module):
    def __init__(self, users: int, items: int, factors: int = 16, hidden: int = 32):
        super().__init__()
        self.gmf_user = nn.Embedding(users, factors)
        self.gmf_item = nn.Embedding(items, factors)
        self.mlp_user = nn.Embedding(users, factors)
        self.mlp_item = nn.Embedding(items, factors)
        self.mlp = nn.Sequential(nn.Linear(2 * factors, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.output = nn.Linear(factors + hidden // 2, 1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor):
        gmf = self.gmf_user(user_ids) * self.gmf_item(item_ids)
        deep = self.mlp(torch.cat((self.mlp_user(user_ids),
                                   self.mlp_item(item_ids)), dim=-1))
        return self.output(torch.cat((gmf, deep), dim=-1)).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = NeuMF(10, 20)
    logits = model(torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]))
    loss = nn.BCEWithLogitsLoss()(logits, torch.tensor([1., 0., 1.]))
    loss.backward()
    print("NCF logits:", tuple(logits.shape), "loss:", round(loss.item(), 4))
