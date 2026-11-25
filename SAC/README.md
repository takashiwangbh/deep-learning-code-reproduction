# SAC：最大熵连续控制

## 1. 出处、提出者与意义

Tuomas Haarnoja、Aurick Zhou、Pieter Abbeel 和 Sergey Levine 于 2018 年提出 [《Soft Actor-Critic》](https://arxiv.org/abs/1801.01290)，随后在 [《Soft Actor-Critic Algorithms and Applications》](https://arxiv.org/abs/1812.05905) 讨论改进。SAC 同时追求高回报与足够随机的动作，既要做得好，也要保持探索。它是离策略方法，可复用回放数据；两套 Q 网络取较小值，缓解价值高估。这里写的是常见的双 Q、随机策略版本，温度 `α` 固定，便于看清核心更新。

## 2. 源码导航与直观解释

[sac.py](sac.py) 的 `GaussianActor` 对每个状态输出高斯均值和标准差；`sample` 先用重参数化采样，再用 `tanh` 把连续动作压到 `[-1,1]`，同时修正动作的对数概率。`QNetwork` 把状态 `[B,S]` 和动作 `[B,A]` 合并，输出 `[B,1]`。`ReplayBuffer` 随机抽取过去的转换；`critic_loss` 计算双 Q 的 Bellman 目标，`actor_loss` 让策略在 Q 高和熵高之间权衡。`update` 先更新两个 critic，再更新 actor，最后 `soft_update` 缓慢跟踪目标 Q 网络。根目录运行 `python SAC/sac.py`，可检查一次随机批次的更新。

完整训练还需真实环境、探索收集、重复调用 `update` 的训练循环和评估。示例中的随机数据不能说明策略学会了控制。

## 3. 数学解释

策略目标是最大化 `E[Q(s,a) - α log π(a|s)]`；`-log π` 对应随机性奖励，所以 actor 损失写为 `E[α log π - min(Q₁,Q₂)]`。双 Q 目标 `y=r+γ(1-d)[min(Q₁'(s',a'),Q₂'(s',a'))-α log π(a'|s')]`，其中 `a'` 由当前策略采样；两个 critic 分别拟合 `y`。动作由 `a=tanh(u)` 得到，因此 `log π(a|s)=log N(u;μ,σ)-Σ log(1-tanh²(u))`；忽略最后一项会把压缩后的动作概率算错。源码用稳定的 `softplus` 等价式计算这个修正。
