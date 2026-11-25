"""TD3 with clipped double Q, target policy smoothing and delayed actor updates."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64,
                 max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action
        self.network = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, action_dim), nn.Tanh())

    def forward(self, states: torch.Tensor):
        return self.network(states) * self.max_action


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        def branch():
            return nn.Sequential(nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.q1, self.q2 = branch(), branch()

    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        pair = torch.cat((states, actions), dim=-1)
        return self.q1(pair), self.q2(pair)


@torch.no_grad()
def smoothed_target_action(target_actor: Actor, next_states: torch.Tensor,
                           noise_std: float = 0.2, noise_clip: float = 0.5):
    action = target_actor(next_states)
    noise = (torch.randn_like(action) * noise_std).clamp(-noise_clip, noise_clip)
    return (action + noise).clamp(-target_actor.max_action, target_actor.max_action)


@torch.no_grad()
def target_value(target_actor: Actor, target_critic: TwinCritic, rewards,
                 next_states, dones, gamma: float = 0.99, noise_std: float = 0.2,
                 noise_clip: float = 0.5):
    action = smoothed_target_action(target_actor, next_states, noise_std, noise_clip)
    q1, q2 = target_critic(next_states, action)
    return rewards + gamma * (1 - dones) * torch.minimum(q1, q2)


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float = 0.005):
    for target_param, online_param in zip(target.parameters(), online.parameters()):
        target_param.lerp_(online_param, tau)


def update(actor: Actor, critic: TwinCritic, target_actor: Actor,
           target_critic: TwinCritic, actor_optimizer, critic_optimizer, batch,
           step: int, policy_delay: int = 2, gamma: float = 0.99, tau: float = 0.005):
    states, actions, rewards, next_states, dones = batch
    desired = target_value(target_actor, target_critic, rewards, next_states, dones, gamma)
    q1, q2 = critic(states, actions)
    critic_loss = F.mse_loss(q1, desired) + F.mse_loss(q2, desired)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    result = {"critic": critic_loss.detach(), "actor": None}
    if step % policy_delay == 0:
        for param in critic.parameters():
            param.requires_grad_(False)
        policy_loss = -critic(states, actor(states))[0].mean()
        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()
        for param in critic.parameters():
            param.requires_grad_(True)
        soft_update(target_actor, actor, tau)
        soft_update(target_critic, critic, tau)
        result["actor"] = policy_loss.detach()
    return result


if __name__ == "__main__":
    torch.manual_seed(0)
    actor, critic = Actor(4, 2), TwinCritic(4, 2)
    target_actor, target_critic = deepcopy(actor), deepcopy(critic)
    a_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    q_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    batch = (torch.randn(6, 4), torch.randn(6, 2).tanh(),
             torch.randn(6, 1), torch.randn(6, 4), torch.zeros(6, 1))
    first = update(actor, critic, target_actor, target_critic, a_opt, q_opt, batch, step=1)
    second = update(actor, critic, target_actor, target_critic, a_opt, q_opt, batch, step=2)
    print("TD3 delayed actor:", first["actor"], "then", round(second["actor"].item(), 4))
