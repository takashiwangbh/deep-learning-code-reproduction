"""Double DQN separates next-action selection from target value evaluation."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, actions: int, hidden: int = 64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, actions))

    def forward(self, states: torch.Tensor):
        return self.network(states)


@torch.no_grad()
def double_dqn_target(online: nn.Module, target: nn.Module,
                      rewards: torch.Tensor, next_states: torch.Tensor,
                      dones: torch.Tensor, gamma: float = 0.99):
    """All rewards and dones are [B]; next_states is [B,state_dim]."""
    next_actions = online(next_states).argmax(dim=-1, keepdim=True)
    next_values = target(next_states).gather(1, next_actions).squeeze(1)
    return rewards + gamma * (1 - dones) * next_values


def q_loss(online: nn.Module, target: nn.Module, batch, gamma: float = 0.99):
    states, actions, rewards, next_states, dones = batch
    predicted = online(states).gather(1, actions.long().unsqueeze(1)).squeeze(1)
    desired = double_dqn_target(online, target, rewards, next_states, dones, gamma)
    return F.smooth_l1_loss(predicted, desired)


@torch.no_grad()
def copy_target(target: nn.Module, online: nn.Module):
    target.load_state_dict(online.state_dict())


if __name__ == "__main__":
    torch.manual_seed(0)
    online = QNetwork(4, 3)
    target = deepcopy(online)
    batch = (torch.randn(6, 4), torch.randint(0, 3, (6,)),
             torch.randn(6), torch.randn(6, 4), torch.zeros(6))
    loss = q_loss(online, target, batch)
    loss.backward()
    print("Double DQN target:", tuple(double_dqn_target(online, target, batch[2],
          batch[3], batch[4]).shape), "loss:", round(loss.item(), 4))
