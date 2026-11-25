# A2C：同步优势演员评论家

## 1. 出处、提出者与意义

Volodymyr Mnih、Adrià Puigdomènech Badia 等人的 [《Asynchronous Methods for Deep Reinforcement Learning》](https://arxiv.org/abs/1602.01783)（2016）提出 A3C：用多个环境采样，actor 选动作，critic 估计状态价值。A2C 是它的**同步**批量版本，由 [OpenAI Baselines](https://openai.com/index/openai-baselines-acktr-a2c/) 推广；这里没有把 A2C 误称为一篇单独的新论文。同步的意思是先收集一批轨迹，再用同一批数据完成一次更新。优势值告诉策略“这个动作比当前状态的平均预期好多少”，能减少只看回报的梯度波动。

## 2. 源码导航与直观解释

打开 [a2c.py](a2c.py)：`ActorCritic` 共享一层状态特征，分别输出动作 logits `[T,B,A]` 与状态价值 `[T,B]`。`n_step_returns` 从轨迹末尾倒着计算折扣回报，遇到 `done=1` 就不再借用下个状态的价值。`a2c_loss` 中策略分支用 `log_prob(action) × advantage`，价值分支拟合回报，熵项鼓励动作分布不要过早只剩一种。根目录运行 `python A2C/a2c.py`，会对小批随机轨迹执行反向传播。

脚本展示一次同步批量更新所需的数学核心；不包括多进程环境采样、真实奖励设计或 benchmark 训练。输入 `[T,B]` 中 `T` 是轨迹时间，`B` 可以代表并行环境。

## 3. 数学解释

从末尾开始的 `n` 步目标递推为 `R_t = r_t + γ(1-d_t)R_{t+1}`，最后用 critic 给的 `V(s_{T})` 启动递推。`d_t=1` 表示终止，阻止回报穿越回合边界。优势估计 `A_t = R_t - V(s_t)`：正值表示动作结果好于原来预计，负值则相反。策略损失 `L_π=-E[log π(a_t|s_t) stopgrad(A_t)]`，critic 损失 `L_V=E[(V(s_t)-stopgrad(R_t))²]`。`stopgrad` 在源码中对应 `.detach()`，让策略更新不会通过优势值偷偷改写 critic。总损失还减去小权重的策略熵 `H(π)`，使探索保留一些余地。
