# TD3：双评论家与延迟策略更新

## 1. 出处、提出者与意义

Scott Fujimoto、Herke van Hoof、David Meger 在 2018 年提出 [《Addressing Function Approximation Error in Actor-Critic Methods》](https://arxiv.org/abs/1802.09477)，算法称 TD3。连续动作方法 DDPG 让 actor 最大化 critic 的估计值，但 critic 的错误高分会诱导 actor 利用估计漏洞。TD3 训练两套 critic，目标取两者较小值；actor 更新频率比 critic 低，并给目标动作加裁剪噪声，避免价值函数在单个尖锐动作附近过拟合。这三点构成算法名称中的 Twin、Delayed 与设计核心。

## 2. 源码导航与直观解释

[td3.py](td3.py) 的 `Actor` 输出受 `tanh` 限制的连续动作 `[B,A]`，`TwinCritic` 分别输出两个 `[B,1]` 的 Q 值。`smoothed_target_action` 在目标 actor 的输出上加裁剪高斯噪声；`target_value` 用目标双 Q 中较小者建立 Bellman 目标。`update` 每次更新 critic，只有 `step % policy_delay == 0` 才更新 actor 和目标网络。根目录运行 `python TD3/td3.py` 可看到第 1 步只更新 critic，第 2 步才有 actor 损失。示例省略真实环境、回放缓冲区和探索采样策略；随机数据仅验证更新逻辑。

## 3. 数学解释

目标动作先算 `a'=clip(μ_target(s')+clip(ε,-c,c),-a_max,a_max)`，其中 `ε∼N(0,σ²)`；这对应代码的双重 `clamp`。两个 critic 的共同训练目标为 `y=r+γ(1-d) min(Q₁_target(s',a'),Q₂_target(s',a'))`。各自最小化 `(Q_i(s,a)-y)²`，最小值限制过高估计。延迟轮次的 actor 损失为 `-E[Q₁(s,μ(s))]`，只用第一套 critic 对 actor 求梯度；目标参数按 `θ_target←(1-τ)θ_target+τθ_online` 缓慢跟随。`no_grad` 保证目标和噪声不参与当前梯度，`policy_delay` 决定两类网络的更新频率。
