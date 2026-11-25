# DenseNet-121：把前面所有层的特征留给后面

## 1. 出处、提出者与意义

Gao Huang、Zhuang Liu、Laurens van der Maaten 和 Kilian Q. Weinberger 在 2016 年提出 [《Densely Connected Convolutional Networks》](https://arxiv.org/abs/1608.06993)。深层网络容易重复学习已有特征，也可能让早层难以获得有效梯度。DenseNet 让每一层直接读到它前面所有层的特征图，促进特征复用和梯度传递。它与 ResNet 的“相加”不同，使用的是沿通道方向“拼接”。

## 2. 源码导航与直观解释

打开 [densenet121.py](densenet121.py)：`DenseLayer` 先用 1×1 卷积整理通道，再用 3×3 卷积产生少量**新**特征；`DenseBlock.forward` 每次把旧特征和新特征拼在一起，下一层可同时使用两者；`Transition` 压缩通道并下采样，避免特征图无限变大。`DenseNet121` 默认四个块层数 `(6,12,24,16)`，输入 `[B,3,H,W]`，最后经平均池化和线性分类器得到 `[B,类别数]` logits。

从仓库根目录运行 `python "DenseNet/densenet121.py"`。底部示例减小增长率和块长度，便于观察形状；默认配置更接近 DenseNet-121，但随机初始化没有分类能力。

## 3. 数学解释

第 `l` 层的输入可以记作 `[x₀,x₁,…,xₗ₋₁]`，它只需学习 `xₗ=Hₗ([x₀,…,xₗ₋₁])`，`Hₗ` 是该层卷积变换，方括号表示通道拼接。源码对应 `torch.cat((x,new_features),dim=1)`。若块开始有 `C₀` 个通道、每层新增 `k` 个通道，那么 `l` 层后通道数为 `C₀+lk`；`DenseBlock.out_channels` 就按此计算。与 `x+F(x)` 不同，拼接保留旧特征原样，后续层能自行选择复用哪些特征；代价是通道持续增加，所以 `Transition` 用 1×1 卷积和平均池化压缩它。
