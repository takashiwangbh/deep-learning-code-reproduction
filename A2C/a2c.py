"""Synchronous advantage actor-critic with bootstrapped n-step returns."""

import torch
from torch import nn
from torch.nn import functional as F


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, actions: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh())
        self.policy = nn.Linear(hidden, actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, states: torch.Tensor):
        hidden = self.trunk(states)
        return self.policy(hidden), self.value(hidden).squeeze(-1)


def n_step_returns(rewards: torch.Tensor, dones: torch.Tensor,
                   last_value: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """[time,batch] returns; done=1 removes bootstrap across episode boundary."""
    result = torch.empty_like(rewards)
    running = last_value.detach()
    for t in reversed(range(rewards.shape[0])):
        running = rewards[t] + gamma * (1 - dones[t]) * running
        result[t] = running
    return result


def a2c_loss(logits: torch.Tensor, values: torch.Tensor,
             actions: torch.Tensor, returns: torch.Tensor,
             value_weight: float = 0.5, entropy_weight: float = 0.01):
    """Policy gradient sees detached advantages; critic fits return targets."""
    policy = torch.distributions.Categorical(logits=logits)
    advantage = returns.detach() - values
    policy_loss = -(policy.log_prob(actions) * advantage.detach()).mean()
    value_loss = F.mse_loss(values, returns.detach())
    entropy = policy.entropy().mean()
    return policy_loss + value_weight * value_loss - entropy_weight * entropy


if __name__ == "__main__":
    torch.manual_seed(0)
    net = ActorCritic(5, 3)
    logits, values = net(torch.randn(4, 2, 5))
    rewards = torch.randn(4, 2)
    dones = torch.zeros_like(rewards)
    dones[-1, 0] = 1
    returns = n_step_returns(rewards, dones, torch.zeros(2))
    loss = a2c_loss(logits, values, torch.randint(0, 3, (4, 2)), returns)
    loss.backward()
    print("A2C logits/value:", tuple(logits.shape), tuple(values.shape))
