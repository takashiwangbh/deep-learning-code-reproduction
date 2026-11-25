# Deep Q-Network（DQN）

## 1. 出处、提出者与意义

Volodymyr Mnih、Koray Kavukcuoglu、David Silver 等人在 [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)（Nature，2015）中展示 DQN：用深度网络从游戏画面估计各个动作的长期价值，再选择价值高的动作。它把“看见图像”与“通过试错学习行动”连起来，是深度强化学习的代表性起点。

训练时相邻经历高度相关，而且模型要用自己不断变化的预测值当学习目标。DQN 的两项关键设计是**经验回放**（打乱并重复使用过去经历）和**目标网络**（让学习目标在一段时间内保持相对稳定）。项目展示这些步骤，不提供 Atari ROM 或论文游戏成绩。

## 2. 源码导航与直观解释

[dqn.py](dqn.py) 中 `AtariQNetwork` 接收连续四帧 `[B,4,84,84]`，输出 `[B,动作数]` 的 Q 值；`SmallQNetwork` 用小状态演示同一算法。`Transition` 保存 `(状态,动作,奖励,下一状态,是否终止)`，`ReplayBuffer` 保存并随机抽取这些经历。`DQNAgent.update` 从在线网络取本次动作的值，从目标网络计算下一步的目标值，求损失后只更新在线网络；`sync_target` 应按训练计划定期同步。

从根目录运行 `python 'DQN/dqn.py'`。示例构造小型回放数据并做一次更新，方便在 CPU 上观察损失。`act` 用 `epsilon` 控制探索：有一定概率随机行动，否则选择估计 Q 值最高的动作。源码同时给出论文风格卷积网络和轻量教学网络，但完整 Atari 数据预处理、环境交互与长期训练需另行接入。

## 3. 数学解释

`Q(s,a)` 表示在状态 `s` 采取动作 `a` 后，长期预计能得到多少奖励。对一条经历 `(s,a,r,s′,done)`，目标网络给出的学习目标是 `y = r + γ(1−done) maxₐ′ Q_target(s′,a′)`，`γ` 是未来奖励的折扣。若这一步已终止，`done=1`，未来项就变成 0。源码中的 `td_target` 对应 `y`，`chosen_q` 是在线网络的 `Q_online(s,a)`。

训练要让 `chosen_q` 靠近 `td_target`；本实现使用平滑 L1 损失，以减少极端误差的影响。`td_target` 放在 `torch.no_grad()` 中，说明它只是本次更新的目标，梯度不会穿过目标网络。回放池随机采样使连续经历不必按产生顺序进入同一更新；定期复制在线网络参数到目标网络，避免每一步都追逐立即移动的目标。
