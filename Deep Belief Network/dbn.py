"""Greedy RBM pretraining and top-down sampling for a small deep belief net."""

import torch
from torch import nn
from torch.nn import functional as F


class BernoulliRBM(nn.Module):
    def __init__(self, visible: int, hidden: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(visible, hidden) * 0.01)
        self.visible_bias = nn.Parameter(torch.zeros(visible))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden))

    def hidden_prob(self, visible):
        return torch.sigmoid(visible @ self.weight + self.hidden_bias)

    def visible_prob(self, hidden):
        return torch.sigmoid(hidden @ self.weight.T + self.visible_bias)

    def free_energy(self, visible):
        return -(visible * self.visible_bias).sum(-1) - F.softplus(
            visible @ self.weight + self.hidden_bias).sum(-1)

    @torch.no_grad()
    def gibbs(self, visible, steps: int = 1):
        for _ in range(steps):
            hidden = torch.bernoulli(self.hidden_prob(visible))
            visible = torch.bernoulli(self.visible_prob(hidden))
        return visible

    def cd_loss(self, data, steps: int = 1):
        negative = self.gibbs(data, steps)
        return self.free_energy(data).mean() - self.free_energy(negative.detach()).mean()


class DeepBeliefNetwork(nn.Module):
    def __init__(self, widths: list[int]):
        super().__init__()
        if len(widths) < 3:
            raise ValueError("DBN needs a visible layer and at least two hidden layers")
        self.rbms = nn.ModuleList(BernoulliRBM(a, b) for a, b in zip(widths[:-1], widths[1:]))

    def encode(self, visible: torch.Tensor):
        for rbm in self.rbms:
            visible = rbm.hidden_prob(visible)
        return visible

    def greedy_pretrain(self, binary_data: torch.Tensor, epochs: int = 3,
                        learning_rate: float = 0.05, cd_steps: int = 1):
        """Train one RBM at a time; sample hidden units as next layer's data."""
        data = binary_data.detach()
        history = []
        for rbm in self.rbms:
            optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate)
            for _ in range(epochs):
                loss = rbm.cd_loss(data, cd_steps)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            history.append(float(loss.detach()))
            with torch.no_grad():
                data = torch.bernoulli(rbm.hidden_prob(data))
        return history

    @torch.no_grad()
    def generate(self, batch_size: int, gibbs_steps: int = 10):
        """Top RBM is undirected; lower RBMs provide directed downward conditionals."""
        top = self.rbms[-1]
        visible = torch.bernoulli(torch.full((batch_size, top.weight.shape[0]), .5,
                                             device=top.weight.device))
        visible = top.gibbs(visible, gibbs_steps)
        for lower in reversed(self.rbms[:-1]):
            visible = torch.bernoulli(lower.visible_prob(visible))
        return visible


if __name__ == "__main__":
    torch.manual_seed(0)
    model = DeepBeliefNetwork([6, 4, 3])
    data = torch.randint(0, 2, (12, 6)).float()
    losses = model.greedy_pretrain(data)
    print("DBN top representation/sample:", tuple(model.encode(data).shape),
          tuple(model.generate(2).shape), "pretraining:", [round(x, 4) for x in losses])
