"""Bernoulli-Bernoulli RBM with Gibbs sampling and CD-k training loss."""

import torch
from torch import nn
from torch.nn import functional as F


class BernoulliRBM(nn.Module):
    def __init__(self, visible: int, hidden: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(visible, hidden) * 0.01)
        self.visible_bias = nn.Parameter(torch.zeros(visible))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden))

    def hidden_prob(self, visible: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(visible @ self.weight + self.hidden_bias)

    def visible_prob(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(hidden @ self.weight.T + self.visible_bias)

    def free_energy(self, visible: torch.Tensor) -> torch.Tensor:
        """-log unnormalized probability of visible state, one value per example."""
        return -(visible * self.visible_bias).sum(-1) - F.softplus(
            visible @ self.weight + self.hidden_bias).sum(-1)

    @torch.no_grad()
    def gibbs(self, visible: torch.Tensor, steps: int = 1) -> torch.Tensor:
        state = visible
        for _ in range(steps):
            hidden = torch.bernoulli(self.hidden_prob(state))
            state = torch.bernoulli(self.visible_prob(hidden))
        return state

    def cd_loss(self, data: torch.Tensor, steps: int = 1):
        """CD-k gradient approximation: data phase minus detached negative phase."""
        if steps < 1:
            raise ValueError("contrastive divergence needs at least one Gibbs step")
        negative = self.gibbs(data, steps)
        return self.free_energy(data).mean() - self.free_energy(negative.detach()).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BernoulliRBM(6, 4)
    data = torch.randint(0, 2, (8, 6)).float()
    loss = model.cd_loss(data)
    loss.backward()
    print("RBM data/hidden:", tuple(data.shape), tuple(model.hidden_prob(data).shape),
          "CD-1:", round(loss.item(), 4))
