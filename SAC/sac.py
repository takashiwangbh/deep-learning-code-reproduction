"""Continuous-action SAC with a squashed Gaussian actor and twin Q critics."""

from collections import deque
import math
import random

import torch
from torch import nn
from torch.nn import functional as F


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def sample(self, state: torch.Tensor):
        hidden = self.trunk(state)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(-5, 2)
        normal = torch.distributions.Normal(mean, log_std.exp())
        raw = normal.rsample()  # reparameterization keeps gradients through the sample
        action = raw.tanh()
        # Change of variables for tanh; use stable log(1-tanh(raw)^2).
        correction = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        log_prob = (normal.log_prob(raw) - correction).sum(dim=-1, keepdim=True)
        return action, log_prob


class QNetwork(nn.Module):
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
def soft_update(target: nn.Module, online: nn.Module, tau: float = 0.005):
    for target_param, online_param in zip(target.parameters(), online.parameters()):
        target_param.lerp_(online_param, tau)


def critic_loss(actor, q1, q2, target_q1, target_q2, batch,
                alpha: float = 0.2, gamma: float = 0.99):
    states, actions, rewards, next_states, dones = batch
    with torch.no_grad():
        next_actions, next_log_prob = actor.sample(next_states)
        next_q = torch.minimum(target_q1(next_states, next_actions),
                               target_q2(next_states, next_actions))
        target = rewards + gamma * (1 - dones) * (next_q - alpha * next_log_prob)
    return F.mse_loss(q1(states, actions), target) + F.mse_loss(q2(states, actions), target)


def actor_loss(actor, q1, q2, states, alpha: float = 0.2):
    actions, log_prob = actor.sample(states)
    return (alpha * log_prob - torch.minimum(q1(states, actions), q2(states, actions))).mean()


def update(actor, q1, q2, target_q1, target_q2, actor_optimizer,
           critic_optimizer, batch, alpha: float = 0.2, gamma: float = 0.99,
           tau: float = 0.005):
    q_loss = critic_loss(actor, q1, q2, target_q1, target_q2, batch, alpha, gamma)
    critic_optimizer.zero_grad()
    q_loss.backward()
    critic_optimizer.step()

    for critic in (q1, q2):
        for param in critic.parameters():
            param.requires_grad_(False)
    policy_loss = actor_loss(actor, q1, q2, batch[0], alpha)
    actor_optimizer.zero_grad()
    policy_loss.backward()
    actor_optimizer.step()
    for critic in (q1, q2):
        for param in critic.parameters():
            param.requires_grad_(True)

    soft_update(target_q1, q1, tau)
    soft_update(target_q2, q2, tau)
    return q_loss.detach(), policy_loss.detach()


if __name__ == "__main__":
    from copy import deepcopy

    torch.manual_seed(0)
    actor, q1, q2 = GaussianActor(4, 2), QNetwork(4, 2), QNetwork(4, 2)
    replay = ReplayBuffer(24)
    for _ in range(8):
        replay.add(torch.randn(4), torch.randn(2).tanh(),
                   torch.randn(()), torch.randn(4), False)
    q_loss, p_loss = update(
        actor, q1, q2, deepcopy(q1), deepcopy(q2),
        torch.optim.Adam(actor.parameters(), lr=1e-3),
        torch.optim.Adam((*q1.parameters(), *q2.parameters()), lr=1e-3),
        replay.sample(6),
    )
    print("SAC critic/actor losses:", round(q_loss.item(), 4), round(p_loss.item(), 4))
