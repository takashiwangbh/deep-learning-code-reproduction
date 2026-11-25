# DeepAR：预测未来值的概率分布

## 1. 出处、提出者与意义

David Salinas、Valentin Flunkert、Jan Gasthaus 在 2017 年 [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110) 中提出用共享循环模型学习大量相关时间序列，并输出未来值的条件分布。相较只给一个点估计，概率预测还能表达不确定性，适合库存、需求等需要风险判断的任务。

## 2. 源码导航与直观解释

[deepar.py](deepar.py) 本地写出 GRU 门控递推；`step` 从上一个实数和隐藏状态预测高斯均值与正标准差；`forward` 使用真实历史值逐步计算条件分布；`negative_log_likelihood` 训练分布参数；`sample` 从分布自回归采样未来。输入 `[B,T]`，预测参数 `[B,T−1]`，未来样本 `[B,H]`。运行 `python DeepAR/deepar.py`。本例只有单变量高斯头，省去原系统的尺度归一化、协变量和非高斯似然。

## 3. 数学解释

模型学习 `p(z_t|z_<t;θ)=N(μ_t,σ_t²)`，其中 `(μ_t,σ_t)` 由循环隐藏状态得到，`σ_t=softplus(a_t)+ε>0`。高斯负对数似然为 `½log(2π)+log σ_t+½((z_t−μ_t)/σ_t)²`，源码逐位置计算并对有效观察值平均。推理时依次采样 `ẑ_t=μ_t+σ_t ε_t`，`ε_t∼N(0,1)`，再把样本送入下一步。训练与采样的输入路径因此不同。

标准差若允许为负数，高斯密度便无意义；代码用 softplus 后再加极小常数，确保任何输入下都为正。若某时间点缺失，observed 参数可将其从损失平均中排除，但这个小例子没有做完整缺失值填补。预测均值是点估计，反复 sample 则得到不同的未来轨迹，可用来估计预测区间。
