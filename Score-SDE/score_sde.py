"""Variance-preserving score SDE: exact forward marginal and reverse Euler sampler."""

import math

import torch
from torch import nn


class VPSDE:
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        if not 0 < beta_min <= beta_max:
            raise ValueError("Require 0 < beta_min <= beta_max")
        self.beta_min, self.beta_max = beta_min, beta_max

    def beta(self, time: torch.Tensor):
        return self.beta_min + time * (self.beta_max - self.beta_min)

    def drift(self, x: torch.Tensor, time: torch.Tensor):
        return -0.5 * self.beta(time)[:, None] * x

    def diffusion(self, time: torch.Tensor):
        return torch.sqrt(self.beta(time))

    def marginal(self, clean: torch.Tensor, time: torch.Tensor, noise: torch.Tensor):
        """Exact x_t distribution for d x = -.5 beta(t) x dt + sqrt(beta(t)) dW."""
        integral = self.beta_min * time + 0.5 * (self.beta_max - self.beta_min) * time.square()
        mean_coefficient = torch.exp(-0.5 * integral)[:, None]
        standard_deviation = torch.sqrt((1 - mean_coefficient.square()).clamp_min(0))
        return mean_coefficient * clean + standard_deviation * noise, standard_deviation


class ScoreNetwork(nn.Module):
    def __init__(self, dimensions: int = 2, hidden: int = 64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimensions + 1, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, dimensions))

    def forward(self, noisy: torch.Tensor, time: torch.Tensor):
        return self.network(torch.cat((noisy, time[:, None]), dim=-1))


def score_matching_loss(network: ScoreNetwork, sde: VPSDE, clean: torch.Tensor,
                        epsilon: float = 1e-3):
    time = torch.rand(clean.shape[0], device=clean.device) * (1 - epsilon) + epsilon
    noise = torch.randn_like(clean)
    noisy, std = sde.marginal(clean, time, noise)
    predicted = network(noisy, time)
    # Gaussian conditional score is -noise/std; multiplying by std balances t.
    return (predicted * std + noise).square().sum(dim=-1).mean()


@torch.no_grad()
def reverse_sample(network: ScoreNetwork, sde: VPSDE, initial_noise: torch.Tensor,
                   steps: int = 100, epsilon: float = 1e-3):
    """Reverse-time Euler-Maruyama from t=1 to epsilon, predictor only."""
    if steps < 1 or not 0 < epsilon < 1:
        raise ValueError("Need steps>=1 and epsilon in (0,1)")
    state = initial_noise.clone()
    step_size = (1 - epsilon) / steps
    for index in range(steps):
        time = torch.full((state.shape[0],), 1 - index * step_size, device=state.device,
                          dtype=state.dtype)
        beta = sde.beta(time)[:, None]
        drift = sde.drift(state, time) - beta * network(state, time)
        state = state - drift * step_size
        state = state + beta.sqrt() * math.sqrt(step_size) * torch.randn_like(state)
    return state


if __name__ == "__main__":
    torch.manual_seed(0)
    network, sde = ScoreNetwork(), VPSDE()
    loss = score_matching_loss(network, sde, torch.randn(8, 2))
    loss.backward()
    sample = reverse_sample(network, sde, torch.randn(3, 2), steps=8)
    print("Score-SDE loss/sample:", round(loss.item(), 4), tuple(sample.shape))
