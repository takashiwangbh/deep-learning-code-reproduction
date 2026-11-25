# Xception：极致分解的卷积网络

## 1. 出处、提出者与意义

François Chollet 在 2016 年 [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) 中将 Inception 理解为从普通卷积走向深度可分离卷积的一步，进一步提出大量使用 depthwise 与 pointwise 分解的 Xception。核心观点是把空间相关性与通道相关性分别建模，再由残差路径帮助深层训练。

## 2. 源码导航与直观解释

[xception.py](xception.py) 的 `SeparableConv` 先按通道做 3×3 空间卷积，再用 1×1 混合通道。`XceptionBlock` 连续使用两次分解卷积，并在需要下采样时让主路和捷径同步变小。`Xception` 仅堆叠几个块，输入 `[B,3,32,32]`，输出分类 logits；运行 `python Xception/xception.py`。这保留了 Xception 的核心算子与捷径，省去论文完整 entry/middle/exit flow 的深度配置。

## 3. 数学解释

对 `M` 输入通道和 `N` 输出通道，普通 3×3 卷积需要约 `9MN` 个权重。分解后，逐通道空间核用 `9M` 个权重，1×1 通道混合用 `MN` 个权重，共 `9M+MN`。代码中的 `groups=in_channels` 实现前半步；之后的 1×1 卷积实现后半步。残差输出 `y=F(x)+P(x)`，若通道或尺寸改变，投影 `P` 也使用相应步长，使相加合法。

Xception 与 MobileNet v1 都使用深度可分离卷积，但 Xception 的论文把它作为接近“极端 Inception”的表达方式，并在更深的带残差骨干里大量使用。代码中的每个 XceptionBlock 都有主路与捷径；若下采样使空间尺寸从 32 变 16，捷径的步长也必须为 2，否则张量无法相加。
