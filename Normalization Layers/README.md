# Normalization Layers（归一化层）

## 1. 出处、提出者与意义

Sergey Ioffe 和 Christian Szegedy 在 [Batch Normalization](https://arxiv.org/abs/1502.03167)（2015）中提出按同一批样本统计特征的方法；Jimmy Lei Ba、Jamie Ryan Kiros 和 Geoffrey E. Hinton 在 [Layer Normalization](https://arxiv.org/abs/1607.06450)（2016）中提出对单个样本自己的特征做统计。两者都先把数值调整到容易处理的尺度，再让网络学习缩放与平移。

关键区别不是“哪一种更高级”，而是**沿哪个维度算均值**。BatchNorm 依赖同一批中的其它样本，推理时需要保存训练期统计值；LayerNorm 独立处理每个样本，常用于序列模型。理解这个维度差异，读 ResNet 中的 BN 与 Transformer 中的 LN 会容易很多。

## 2. 源码导航与直观解释

[normalization.py](normalization.py) 直接用张量运算实现 `BatchNorm1dFromScratch` 和 `LayerNormFromScratch`。前者输入 `[B,D]`：`B` 是批大小，`D` 是特征数；训练时沿 `B` 求均值与方差，推理时读取 `running_mean/running_var`。后者接受 `[...,D]`，沿最后一维求统计值，因此一个 token 或样本不会读取别的样本的数值。两者都把可学习的 `weight`、`bias` 放在归一化之后。

从仓库根目录运行 `python 'Normalization Layers/normalization.py'`。例子执行前向与反向传播，并确认输入能收到梯度。打开源码时重点比较两行 `mean(dim=0)` 与 `mean(dim=-1, keepdim=True)`；它们决定了两种算法的行为。

## 3. 数学解释

两种方法共用公式 `y = γ (x−μ)/√(σ²+ε) + β`。这里 `μ` 是均值，`σ²` 是方差，`ε` 防止分母过小，`γ` 与 `β` 是模型学出的缩放、平移。对于 `[B,D]` 张量，BatchNorm 的第 `d` 个特征计算 `μ_d=(1/B)Σ_b x_{b,d}`；例如四张图同一个通道的值共同决定这一通道的统计量。训练时的统计值会以移动平均写入 `running_mean/running_var`，供推理使用。

LayerNorm 则对每个样本 `b` 计算 `μ_b=(1/D)Σ_d x_{b,d}`，方差也只用该样本的 `D` 个特征。因此把 batch 中其它样本换掉，不会改变这个样本的 LayerNorm 结果。源码中的平方差均值对应 `σ²`，`torch.rsqrt` 直接算 `1/√(σ²+ε)`；最后乘 `weight`、加 `bias` 对应 `γ`、`β`。
