# RealNVP：可逆变换与精确概率密度

## 1. 出处、提出者与意义

Laurent Dinh、Jascha Sohl-Dickstein 和 Samy Bengio 在 2016 年提出、ICLR 2017 发表 [《Density estimation using Real NVP》](https://arxiv.org/abs/1605.08803)。普通生成器容易采样，却通常不能精确计算某张图的概率；RealNVP 把简单高斯经过一系列可逆变换，既能生成，又能通过变量替换公式精确求对数密度。关键是设计 Jacobian 行列式容易计算的仿射耦合层。

## 2. 源码导航与直观解释

在 [realnvp.py](realnvp.py) 先看 `AffineCoupling`：掩码固定半边向量，让网络只据此预测另一半的缩放与平移。被固定的半边不变，因而逆变换能先读到同样的条件，再撤销缩放和平移。`RealNVP` 交替使用两种掩码，让每个维度都被更新；`forward` 同时累积每层的 log determinant，`inverse` 从后往前撤销，`log_prob` 加上高斯密度。输入输出均是 `[B,4]`。运行 `python RealNVP/realnvp.py` 会检查往返误差。这里用小向量展示核心；论文的多尺度图像结构和 checkerboard 掩码没有复现。

## 3. 数学解释

把输入分成固定部分 `x_a` 与更新部分 `x_b`，耦合层计算 `z_a=x_a`、`z_b=x_b⊙exp(s(x_a))+t(x_a)`；逆式是 `x_b=(z_b−t(z_a))⊙exp(−s(z_a))`。Jacobian 是三角形，因此 `log|det J|=Σ s_i(x_a)`，源码在 `forward` 返回 `scale.sum(-1)`。多层变换 `z=f(x)` 的对数密度是 `log p_X(x)=log p_Z(z)+Σ_l log|det J_l|`，`log_prob` 正对应此式。代码把 `s` 经过 `tanh` 限幅，仅为教学计算稳定；这种限幅不是 RealNVP 理论所必需。
