"""Inverted dropout: sample Bernoulli masks during training, identity at evaluation."""

import torch
from torch import nn
from torch.nn import functional as F


def inverted_dropout(x: torch.Tensor, probability: float, training: bool) -> torch.Tensor:
    if not 0 <= probability < 1:
        raise ValueError("drop probability must be in [0,1)")
    if not training or probability == 0:
        return x
    keep = (torch.rand_like(x) >= probability).to(x.dtype)
    return x * keep / (1 - probability)


class DropoutMLP(nn.Module):
    def __init__(self, features: int, hidden: int, classes: int, probability: float = 0.5):
        super().__init__()
        self.first = nn.Linear(features, hidden)
        self.second = nn.Linear(hidden, classes)
        self.probability = probability

    def forward(self, x: torch.Tensor):
        hidden = F.relu(self.first(x))
        hidden = inverted_dropout(hidden, self.probability, self.training)
        return self.second(hidden)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = DropoutMLP(5, 8, 3)
    train_logits = model(torch.randn(4, 5))
    train_logits.sum().backward()
    model.eval()
    print("Train/eval logits:", tuple(train_logits.shape),
          tuple(model(torch.randn(4, 5)).shape))
