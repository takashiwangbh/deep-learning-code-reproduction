# 05 · 深度强化学习

| 文件 | 关键源码 | 适用场景 |
| --- | --- | --- |
| `dqn.py` | `ReplayBuffer`、`AtariQNetwork`、在线/目标网络、TD 目标 | 离散动作，基于价值学习 |
| `ppo.py` | `ActorCritic`、GAE、裁剪策略损失、多轮更新 | 离散动作，基于策略学习 |

从仓库根目录运行：

```bash
python Deep-Learning-Sourcebook/reinforcement/dqn.py
python Deep-Learning-Sourcebook/reinforcement/ppo.py
python Deep-Learning-Sourcebook/reinforcement/smoke_test.py
```

`DQNAgent.update` 用 `r + γ(1−done) maxₐ Q_target(s′,a)` 作为目标；目标网络不参与梯度更新，需定期 `sync_target()`。`AtariQNetwork` 保留论文中的卷积层布局；入口演示用较小的状态网络进行一次更新，不需要 Atari ROM。

`generalized_advantage_estimate` 从后往前计算 `δₜ=rₜ+γV(sₜ₊₁)−V(sₜ)` 和 `Aₜ=δₜ+γλAₜ₊₁`，终止位置切断后续值。`clipped_policy_loss` 用新旧策略概率比的裁剪区间限制一次更新的变化。PPO 示例演示更新算法，不附带游戏环境和基准分数。

论文：[DQN](https://www.nature.com/articles/nature14236) · [PPO](https://arxiv.org/abs/1707.06347)
