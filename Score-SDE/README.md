# Score-SDE：用连续时间的噪声与反向随机微分方程生成数据

## 1. 出处、提出者与意义

Yang Song、Jascha Sohl-Dickstein、Diederik P. Kingma、Abhishek Kumar、Stefano Ermon、Ben Poole 在 2020 年提出 [《Score-Based Generative Modeling through Stochastic Differential Equations》](https://arxiv.org/abs/2011.13456)。DDPM 使用离散时间步反复加噪、去噪；Score-SDE 把时间视为连续变量，前向 SDE 将数据逐渐变成已知噪声分布，反向 SDE 在学习到的 score 帮助下把噪声拉回数据。这个框架把扩散模型、score matching 和连续采样器联系起来，是后来大量生成方法的重要理论基础。

## 2. 源码导航与直观解释

[score_sde.py](score_sde.py) 的 `VPSDE` 定义连续噪声强度 `beta(t)`、前向漂移和扩散系数；`marginal` 不必模拟所有中间步，就能直接从干净样本得到任意时间的带噪样本。`ScoreNetwork` 从 `[B,D]` 的噪声向量与 `[B]` 时间预测 score。`score_matching_loss` 用已知高斯扰动监督网络；`reverse_sample` 从标准高斯出发，按反向 SDE 的漂移做 Euler-Maruyama 小步。运行 `python Score-SDE/score_sde.py`，可检查随机二维数据的一次梯度和采样。这里是向量数据与 VP-SDE 的预测器版本，未实现论文的图像 U-Net、corrector 步、概率流 ODE 或论文指标。

## 3. 数学解释

前向 VP-SDE 为 `dx=-½β(t)x dt+√β(t)dW`。积分噪声日程后，任意时刻可写为 `x_t=m(t)x₀+√(1-m(t)²)ε`，`m(t)=exp[-½∫₀ᵗ β(s)ds]`；源码 `marginal` 用线性 `β(t)` 的解析积分直接计算。条件高斯的 score 是 `∇_{x_t}log p(x_t|x₀)=-ε/σ(t)`，所以训练损失 `E‖σ(t)sθ(x_t,t)+ε‖²`。反向时间 SDE 的漂移是 `f(x,t)-g(t)²sθ(x,t)`；`reverse_sample` 从 `t=1` 向小时间积分，由于时间步为负，更新写成 `x←x-[f-g²sθ]Δt+g√Δt ξ`。若网络没训练，输出当然不能代表真实数据分布。
