# DDPG：连续动作的确定性策略梯度

## 1. 出处、提出者与意义

Timothy P. Lillicrap、Jonathan J. Hunt 等人在 2015 年的 [《Continuous control with deep reinforcement learning》](https://arxiv.org/abs/1509.02971) 提出 DDPG。DQN 可以给少数离散动作逐个打分，但机器人关节的角度或油门是连续数值，无法逐个枚举。DDPG 用 actor 直接输出连续动作，用 critic 评估“在当前状态做这个动作”有多好。经验回放打散相邻样本的相关性；目标网络慢慢跟随在线网络，避免目标值每一步剧烈变化。这是从离散动作 Q 学习走向连续控制的一项重要工作。

## 2. 源码导航与直观解释

[ddpg.py](ddpg.py) 中的 `Actor` 把状态 `[B,state_dim]` 映射到受 `tanh` 限制的连续动作 `[B,action_dim]`；`Critic` 将状态与动作拼接，输出 `[B,1]` 的 Q 值。`ReplayBuffer` 存储 `(s,a,r,s',done)` 并随机抽样。读 `update` 时先看目标 critic 如何给旧经验设置学习目标，再看 actor 怎样调整动作以获得更高 Q，最后看 `soft_update` 如何缓慢更新两套目标网络。根目录运行 `python DDPG/ddpg.py`，脚本使用随机数据演示一次参数更新。

本例实现核心更新、回放和目标网络；没有真实环境、探索噪声进程、图像编码器或论文训练规模。随机损失值只说明计算路径可运行，不是控制成绩。

## 3. 数学解释

critic 学习的目标是 `y = r + γ(1-done) Q_target(s', μ_target(s'))`。到达终止状态时 `done=1`，未来价值项被去掉；源码的 `with torch.no_grad()` 保证目标网络不会被这一步误更新。critic 最小化 `(Q(s,a)-y)²`；actor 最小化 `-Q(s, μ(s))`，也就是沿 critic 给出的方向改变动作。这里的确定性策略梯度通过 `actor(states)` 和 `critic(states, ...)` 的计算图自动传递。目标参数以 `θ_target ← (1-τ)θ_target + τθ_online` 更新，`tau` 很小时目标变化缓慢。
