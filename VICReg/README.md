# VICReg：显式阻止表示塌缩

## 1. 出处、提出者与意义

Adrien Bardes、Jean Ponce、Yann LeCun 于 2021 年提出 [《VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning》](https://arxiv.org/abs/2105.04906)。无标签训练希望同一图片的两种增强拥有相似表示，但如果所有图片都输出常量，也会满足“相似”。VICReg 分别约束三件事：配对视图应一致、每个特征维度在 batch 内要有足够变化、不同维度不要重复表达同一信息。各项都有直接的数学意义，不需要负样本或非对称教师网络。

## 2. 源码导航与直观解释

[vicreg.py](vicreg.py) 中 `Encoder` 用同一网络处理两种图像视图，输出 `[B,D]`。`vicreg_loss` 首先算两个表示的均方误差；`variance_penalty` 惩罚标准差小于 1 的特征列；`covariance_penalty` 只惩罚协方差矩阵的非对角项。根目录运行 `python VICReg/vicreg.py`，可在 CPU 上完成一次随机样本的反向传播。实际自监督训练需要图像增强、合适的批次和训练日程；随机输入只用于检查损失分支。

## 3. 数学解释

两视图表示分别记为 `Z,Z'∈R^{B×D}`。不变性项 `s=(1/BD)‖Z-Z'‖²` 让对应样本靠近。方差项 `v(Z)=(1/D)Σ_j max(0,1-√(Var(Z_j)+ε))` 强迫每一列跨样本保持变化；只靠它无法保证不同列不重复。协方差项先算中心化 `C=(Z-mean(Z))ᵀ(Z-mean(Z))/(B-1)`，再对 `i≠j` 的 `C_ij²` 求和，削弱冗余。总损失 `L=λs+μ[v(Z)+v(Z')]/2+ν[c(Z)+c(Z')]`；源码让三种权重显式可调。常量输出的方差为零，会受到方差项惩罚。
