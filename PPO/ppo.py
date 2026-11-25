"""离散动作 PPO：策略/价值网络、GAE、裁剪目标、多轮参数更新。"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, num_actions: int, hidden: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(observation_dim, hidden), nn.Tanh(),
                                      nn.Linear(hidden, hidden), nn.Tanh())
        self.policy_head = nn.Linear(hidden, num_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, observations: torch.Tensor):
        features = self.backbone(observations)
        distribution = Categorical(logits=self.policy_head(features))
        values = self.value_head(features).squeeze(-1)
        return distribution, values


def generalized_advantage_estimate(rewards: torch.Tensor, values: torch.Tensor,
                                   dones: torch.Tensor, last_value: torch.Tensor,
                                   gamma: float = 0.99, gae_lambda: float = 0.95):
    """输入按时间排序的 [T] 张量；last_value 是末状态 V(s_T)。"""
    if not (rewards.shape == values.shape == dones.shape):
        raise ValueError("rewards, values and dones must have the same shape")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(len(rewards))):
        alive = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * alive * next_value - values[t]
        gae = delta + gamma * gae_lambda * alive * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def clipped_policy_loss(new_logp: torch.Tensor, old_logp: torch.Tensor,
                        advantages: torch.Tensor, clip_epsilon: float = 0.2):
    ratio = (new_logp - old_logp).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages
    return -torch.minimum(unclipped, clipped).mean()


def ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer,
               observations: torch.Tensor, actions: torch.Tensor,
               old_logp: torch.Tensor, advantages: torch.Tensor,
               returns: torch.Tensor, epochs: int = 4,
               clip_epsilon: float = 0.2, value_coefficient: float = 0.5,
               entropy_coefficient: float = 0.01) -> float:
    """rollout 收集结束后执行；每轮复用旧策略产生的一批样本。"""
    old_logp = old_logp.detach()
    advantages = advantages.detach()
    returns = returns.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    for _ in range(epochs):
        distribution, values = model(observations)
        new_logp = distribution.log_prob(actions)
        policy_loss = clipped_policy_loss(new_logp, old_logp, advantages, clip_epsilon)
        value_loss = F.mse_loss(values, returns)
        entropy = distribution.entropy().mean()
        loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss.item())


if __name__ == "__main__":
    torch.manual_seed(7)
    model = ActorCritic(observation_dim=4, num_actions=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    observations = torch.randn(16, 4)
    with torch.no_grad():
        old_distribution, values = model(observations)
        actions = old_distribution.sample()
        old_logp = old_distribution.log_prob(actions)
        advantages, returns = generalized_advantage_estimate(
            torch.randn(16), values, torch.zeros(16), torch.tensor(0.0)
        )
    print("PPO loss:", round(ppo_update(model, optimizer, observations, actions,
                                       old_logp, advantages, returns), 4))
