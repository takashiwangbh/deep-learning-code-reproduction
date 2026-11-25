# REINFORCE：从完整回报学习随机策略

## 1. 出处、提出者与意义

Ronald J. Williams 在 1992 年的 [《Simple statistical gradient-following algorithms for connectionist reinforcement learning》](https://doi.org/10.1007/BF00992696) 给出了 REINFORCE 系列梯度估计。它直接调节“在某状态选择某动作”的概率：一条轨迹后来获得高回报，就提高当时所选动作的概率；回报差则降低。它不需要先训练 Q 网络，是理解策略梯度、actor-critic 和 PPO 的起点。原始方法方差较大，这也是后续价值基线与优势函数发展的动机。

## 2. 源码导航与直观解释

[reinforce.py](reinforce.py) 的 `Policy` 把状态 `[state_dim]` 映射为离散动作 logits；`act` 通过 `Categorical` 采样动作并保存其 `log_prob`。一回合结束后，`discounted_returns` 从最后一步倒推每个时刻的后续回报；`reinforce_loss` 用回报加权每个已选动作的对数概率。根目录运行 `python REINFORCE/reinforce.py`，会用随机状态和奖励完成一次反向传播。示例没有环境与真实采样轨迹，只展示更新公式；真正训练需从当前策略与环境交互得到奖励。

## 3. 数学解释

令 `πθ(a_t|s_t)` 是策略选择动作的概率，`G_t=Σ_{k=t}^{T-1} γ^{k-t}r_k` 是从第 `t` 步起的折扣回报。策略梯度估计为 `∇θJ≈Σ_t G_t∇θ log πθ(a_t|s_t)`。因此代码最小化 `L=-mean_t[G_t log πθ(a_t|s_t)]`：最小化这个负号损失就对应增大期望回报。`discounted_returns` 的倒序递推 `G_t=r_t+γG_{t+1}` 比每个时间点重新求和简单。回报由环境产生，不允许沿奖励本身反向传播，所以源码对回报使用 `.detach()`；策略概率的梯度仍完整保留。
