"""Second-order Factorization Machine for sparse or dense real-valued features."""

import torch
from torch import nn


class FactorizationMachine(nn.Module):
    def __init__(self, features: int, factors: int = 8):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.linear = nn.Parameter(torch.zeros(features))
        self.factors = nn.Parameter(torch.randn(features, factors) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Real-valued features [B,F] -> scalar logits [B]."""
        if x.ndim != 2 or x.shape[1] != self.linear.numel():
            raise ValueError("Expected a [batch, features] matrix")
        first_order = x @ self.linear
        summed = x @ self.factors
        squared_sum = x.square() @ self.factors.square()
        pairwise = 0.5 * (summed.square() - squared_sum).sum(dim=-1)
        return self.bias + first_order + pairwise


if __name__ == "__main__":
    torch.manual_seed(0)
    model = FactorizationMachine(features=6, factors=4)
    features = torch.tensor([[1., 0., 1., 0., 0., 0.],
                             [0., 1., 0., 1., 0., 1.]])
    labels = torch.tensor([1., 0.])
    logits = model(features)
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()
    print("FM logits:", tuple(logits.shape), "loss:", round(loss.item(), 4))
