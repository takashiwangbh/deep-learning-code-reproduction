# Flow Matching：学习从噪声走向数据的速度

## 1. 出处、提出者与意义

Yaron Lipman、Ricky T. Q. Chen、Heli Ben-Hamu、Maximilian Nickel 与 Matt Le 于 2022 年提出 [《Flow Matching for Generative Modeling》](https://arxiv.org/abs/2210.02747)。它提供一种训练连续生成流的方法：不必在训练时反复求解完整轨迹，而是选定一条概率路径，直接监督每个时间位置的速度场。采样时再沿学到的速度走，从简单噪声到复杂数据。本目录选择最易看懂的直线插值和 Euler 积分，展示这个思想的最小可运行实现；论文讨论的其它概率路径与高效求解器不在此例中。

## 2. 源码导航与直观解释

主源码：[flow_matching.py](flow_matching.py)；时间条件 `TinyUNet` 的完整源码在本项目的 [ddpm.py](ddpm.py)，所以无需安装或导入其它项目的模型。`FlowMatching.interpolation` 给定一张随机噪声图、一张真实图和时间 `t`，返回中间图与应学习的速度；`training_loss` 随机取 `t`，让网络回归这个速度；`sample` 从噪声开始做若干 Euler 小步。输入和输出均为 `[B,C,H,W]`，`t` 为 `[B]`。直观地看，把噪声放在路的起点、真实图放在终点，网络学习“在路上每处往哪走、走多快”。根目录运行 `python 'Flow Matching/flow_matching.py'`。

## 3. 数学解释

记起点噪声为 `x₀∼N(0,I)`、终点样本为 `x₁∼p_data`。直线路径定义 `x_t=(1−t)x₀+t x₁`，其中 `t∈[0,1]`；对 `t` 求导得目标速度 `u_t=dx_t/dt=x₁−x₀`，正是 `interpolation` 返回的第二个量。训练 `v_θ` 最小化 `E‖v_θ(x_t,t)−u_t‖²`，见 `training_loss`。采样时用 Euler 近似微分方程 `dx/dt=v_θ(x,t)`：`x_{t+Δt}=x_t+Δt·v_θ(x_t,t)`，源码 `sample` 将 `Δt` 取为 `1/steps`。步数越少计算越省，但离散误差通常更大；未训练的网络无法凭这条公式生成有意义的图像。
