"""DQN：经验回放、独立目标网络、Bellman TD 更新。

AtariQNetwork 展示论文的卷积架构；脚本入口用小型 MLP 和随机回放演示一次更新，
因此无需安装游戏环境，也不声称得到 Atari 成绩。
"""

from collections import deque
from dataclasses import dataclass
import random

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.memory: deque[Transition] = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(Transition(torch.as_tensor(state, dtype=torch.float32).detach().clone(),
                                      int(action), float(reward),
                                      torch.as_tensor(next_state, dtype=torch.float32).detach().clone(),
                                      bool(done)))

    def sample(self, batch_size: int, device: torch.device | str = "cpu"):
        transitions = random.sample(self.memory, batch_size)
        states = torch.stack([t.state for t in transitions]).to(device)
        actions = torch.tensor([t.action for t in transitions], device=device)
        rewards = torch.tensor([t.reward for t in transitions], device=device)
        next_states = torch.stack([t.next_state for t in transitions]).to(device)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class AtariQNetwork(nn.Module):
    """输入 [B,4,84,84] 的连续四帧，输出每个离散动作的 Q 值。"""

    def __init__(self, num_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
                                  nn.Linear(512, num_actions))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(frames))


class SmallQNetwork(nn.Module):
    """同一算法的小状态版本，方便在 CPU 上学习更新步骤。"""

    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 32), nn.ReLU(),
                                 nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, num_actions))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DQNAgent:
    def __init__(self, online: nn.Module, target: nn.Module, learning_rate: float = 1e-3,
                 gamma: float = 0.99):
        self.online, self.target, self.gamma = online, target, gamma
        self.optimizer = torch.optim.Adam(online.parameters(), lr=learning_rate)
        self.sync_target()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    def sync_target(self):
        """每隔若干训练步调用；目标网络在两次同步之间保持不变。"""
        self.target.load_state_dict(self.online.state_dict())

    def act(self, state: torch.Tensor, epsilon: float = 0.1) -> int:
        if random.random() < epsilon:
            return random.randrange(self.online(state[None]).size(-1))
        with torch.no_grad():
            return int(self.online(state[None]).argmax(dim=-1).item())

    def update(self, batch) -> float:
        states, actions, rewards, next_states, dones = batch
        chosen_q = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1).values
            td_target = rewards + self.gamma * (1 - dones) * next_q
        loss = F.smooth_l1_loss(chosen_q, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


if __name__ == "__main__":
    torch.manual_seed(7)
    agent = DQNAgent(SmallQNetwork(5, 2), SmallQNetwork(5, 2))
    replay = ReplayBuffer()
    for i in range(32):
        state = F.one_hot(torch.tensor(i % 5), 5).float()
        next_state = F.one_hot(torch.tensor((i + 1) % 5), 5).float()
        replay.add(state, i % 2, float(i % 3 == 0), next_state, i % 7 == 0)
    print("DQN TD loss:", round(agent.update(replay.sample(16)), 4))
