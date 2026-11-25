# Glow：用可逆卷积混合通道

## 1. 出处、提出者与意义

Diederik P. Kingma 与 Prafulla Dhariwal 在 2018 年 [《Glow: Generative Flow with Invertible 1x1 Convolutions》](https://arxiv.org/abs/1807.03039) 中把可逆 1×1 卷积放进流模型。它像一个可学习的通道混合矩阵，取代固定通道置换；配合 ActNorm 和仿射耦合，每一步都可逆且能精确计算概率密度。Glow 是理解现代 normalizing flow 的经典例子。

## 2. 源码导航与直观解释

[glow.py](glow.py) 的 `squeeze2d` 把 `[B,1,8,8]` 重排成 `[B,4,4,4]`，没有丢像素；每个 `GlowStep` 再依次做 `ActNorm`、`Invertible1x1Conv`、`AffineCoupling`。`inverse` 严格倒序撤销这三步，`log_prob` 累加高斯潜变量密度与各步 log determinant。运行 `python Glow/glow.py` 会做前向、求密度、反向传播和可逆性断言。这里是单尺度小图版本；论文完整系统还包括多尺度拆分和更大的网络，未训练的随机模型不会生成有意义的图像。

## 3. 数学解释

流模型使用变量替换 `log p_X(x)=log p_Z(f(x))+log|det ∂f/∂x|`。`squeeze` 只是元素重排，行列式绝对值为 1，所以贡献 0。ActNorm 按通道计算 `y=(x+b)exp(s)`，在 `H×W` 个空间位置重复，故贡献 `HW·Σ_c s_c`。1×1 卷积相当于每个像素乘同一可逆矩阵 `W`，贡献 `HW·log|det W|`；源码用 `torch.linalg.slogdet`。仿射耦合固定一半通道，只对另一半做 `y_b=x_b exp(s(x_a))+t(x_a)`，贡献所有可变位置的 `s` 之和。把这些可计算的量相加，就得到不需要近似下界的精确 log likelihood。
