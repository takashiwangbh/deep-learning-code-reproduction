# Double DQN：选动作和估价值分开

## 1. 出处、提出者与意义

Hado van Hasselt、Arthur Guez、David Silver 在 2015 年提出 [《Deep Reinforcement Learning with Double Q-learning》](https://arxiv.org/abs/1509.06461)。标准 DQN 在下一状态里先找最大的 Q 值，再用同一个值作为学习目标；当多个估计都带噪声时，挑最大值容易选中“高估得最厉害”的动作。Double DQN 让在线网络负责选动作、目标网络负责给这个动作估价，减少这种最大值偏差。这是一项简单但影响深远的 DQN 改进。

## 2. 源码导航与直观解释

[double_dqn.py](double_dqn.py) 的 `QNetwork` 对状态 `[B,S]` 输出每个离散动作的 Q 值 `[B,A]`。`double_dqn_target` 先用 `online(next_states).argmax` 选下一个动作，再用 `target(next_states).gather` 取该动作的分数；到达终止状态时不再追加未来价值。`q_loss` 计算当前所选动作的预测值与这个目标的平滑 L1 误差，`copy_target` 展示定期同步目标网络。根目录运行 `python 'Double DQN/double_dqn.py'`。完整 Atari 训练还需经验回放、探索率日程、图像编码器和大量环境步骤；这里将决定性的目标公式单独展开。

## 3. 数学解释

标准 DQN 目标为 `y=r+γ(1-d) max_a Q_target(s',a)`。Double DQN 改为 `a*=argmax_a Q_online(s',a)`，`y=r+γ(1-d)Q_target(s',a*)`。注意目标网络仍给数值，在线网络只给动作索引；若把两处都换成同一网络，就失去“分开”的关键。源码用 `no_grad` 计算目标，避免误差反向修改挑动作和估价值的网络。`d=1` 代表该样本回合已经结束，此时目标只剩即时奖励。`smooth_l1_loss` 让极大目标误差不至于产生过大的梯度。
