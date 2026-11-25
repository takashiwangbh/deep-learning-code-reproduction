"""Williams REINFORCE: Monte Carlo policy gradient for discrete actions."""

import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self, state_dim: int, actions: int, hidden: int = 32):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(),
                                     nn.Linear(hidden, actions))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)

    def act(self, state: torch.Tensor):
        distribution = torch.distributions.Categorical(logits=self(state))
        action = distribution.sample()
        return action, distribution.log_prob(action)


def discounted_returns(rewards: torch.Tensor, gamma: float = 0.99):
    """One completed episode's reward-to-go; no bootstrap or target network."""
    result = torch.empty_like(rewards)
    running = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for step in range(len(rewards) - 1, -1, -1):
        running = rewards[step] + gamma * running
        result[step] = running
    return result


def reinforce_loss(log_probs: torch.Tensor, rewards: torch.Tensor, gamma: float = 0.99):
    returns = discounted_returns(rewards, gamma)
    return -(log_probs * returns.detach()).sum() / len(rewards)


if __name__ == "__main__":
    torch.manual_seed(0)
    policy = Policy(4, 2)
    log_probs = []
    for state in torch.randn(5, 4):
        _, log_prob = policy.act(state)
        log_probs.append(log_prob)
    loss = reinforce_loss(torch.stack(log_probs), torch.tensor([0., 0., 1., 0., 2.]))
    loss.backward()
    print("REINFORCE loss:", round(loss.item(), 4), "gradient:",
          round(policy.network[0].weight.grad.norm().item(), 4))
