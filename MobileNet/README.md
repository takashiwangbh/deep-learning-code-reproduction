# MobileNet v1：深度可分离卷积

## 1. 出处、提出者与意义

Andrew G. Howard 等人在 2017 年 [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) 中提出适合移动设备的轻量 CNN。最关键的替换是将普通卷积分成每个通道独立的空间卷积与一次跨通道的 1×1 卷积，从而大幅减少计算。它与仓库中另有的 MobileNetV2 倒残差版本是不同项目。

## 2. 源码导航与直观解释

[mobilenet.py](mobilenet.py) 的 `DepthwiseSeparable` 先用 `groups=in_channels` 做 3×3 depthwise 卷积，再用 1×1 pointwise 卷积混合通道；两步各配归一化和 ReLU。`MobileNet` 堆叠这些模块，池化后分类。运行 `python MobileNet/mobilenet.py`，输入 `[B,3,32,32]`，输出 `[B,类别数]`。这里用小通道与少量块演示 v1 的核心，并未实现论文全部层数与宽度/分辨率搜索。

## 3. 数学解释

普通 `k×k` 卷积从 `M` 通道到 `N` 通道的计算量约为 `k²MN·HW`。深度可分离卷积为 `k²M·HW + MN·HW`：第一项逐通道做空间滤波，第二项用 1×1 混合通道。两者之比约为 `1/N+1/k²`，当 `k=3` 且 `N` 较大时显著更小。源码中的 `groups=in_channels` 让每个输入通道拥有自己的空间核。

深度卷积的 groups 参数必须等于输入通道数；这样一个 3×3 核只处理一个通道。其后 1×1 卷积再把这些独立空间特征组合成新通道。如果只用深度卷积而没有 pointwise 步骤，不同通道之间几乎不能交换信息。宽度缩放和输入分辨率也会影响真实设备延迟，不能只凭参数量判断速度。
