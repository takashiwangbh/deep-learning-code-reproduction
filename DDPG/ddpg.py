"""DDPG: deterministic actor, Q critic, replay and slowly updated targets."""

from collections import deque
from copy import deepcopy
import random

import torch
from torch import nn
from torch.nn import functional as F


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0, hidden: int = 64):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, action_dim), nn.Tanh())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((state, action), dim=-1))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.data.append((state.detach().clone(), action.detach().clone(),
                          float(reward), next_state.detach().clone(), float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.data, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states), torch.stack(actions),
                torch.tensor(rewards, dtype=states[0].dtype).unsqueeze(-1),
                torch.stack(next_states),
                torch.tensor(dones, dtype=states[0].dtype).unsqueeze(-1))


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    for target_param, online_param in zip(target.parameters(), online.parameters()):
        target_param.lerp_(online_param, tau)


def update(actor, critic, target_actor, target_critic, actor_optimizer,
           critic_optimizer, batch, gamma: float = 0.99, tau: float = 0.005):
    states, actions, rewards, next_states, dones = batch
    with torch.no_grad():
        next_actions = target_actor(next_states)
        target = rewards + gamma * (1 - dones) * target_critic(next_states, next_actions)

    critic_loss = F.mse_loss(critic(states, actions), target)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    for param in critic.parameters():
        param.requires_grad_(False)
    actor_loss = -critic(states, actor(states)).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    for param in critic.parameters():
        param.requires_grad_(True)

    soft_update(target_actor, actor, tau)
    soft_update(target_critic, critic, tau)
    return critic_loss.detach(), actor_loss.detach()


if __name__ == "__main__":
    torch.manual_seed(0)
    actor, critic = Actor(4, 2), Critic(4, 2)
    target_actor, target_critic = deepcopy(actor), deepcopy(critic)
    replay = ReplayBuffer(32)
    for _ in range(12):
        replay.add(torch.randn(4), torch.randn(2).clamp(-1, 1),
                   torch.randn(()), torch.randn(4), False)
    losses = update(actor, critic, target_actor, target_critic,
                    torch.optim.Adam(actor.parameters(), lr=1e-3),
                    torch.optim.Adam(critic.parameters(), lr=1e-3), replay.sample(8))
    print("DDPG critic/actor losses:", tuple(round(x.item(), 4) for x in losses))
