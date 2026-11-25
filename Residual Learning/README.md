# Residual Learning：先理解残差块

## 1. 出处、提出者与意义

Kaiming He、Xiangyu Zhang、Shaoqing Ren、Jian Sun 在 2015 年 [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) 中提出残差学习。观察到网络继续加深时，训练误差也可能变坏，说明深层优化本身有难度。残差块让卷积路径只学习对输入的修正，并提供直接的捷径；完整 ResNet-34 结构见仓库里的 Residual Network 项目。

## 2. 源码导航与直观解释

[residual_learning.py](residual_learning.py) 的 `ResidualBlock` 有 `transform` 与 `shortcut` 两条路径。`forward` 将二者相加再激活。输入输出通道与尺寸相同时，捷径是原样 `Identity`；若 `stride=2` 或通道变化，则用 1×1 卷积投影。运行 `python 'Residual Learning/residual_learning.py'`，可看到 `[B,8,16,16]` 变成 `[B,16,8,8]`，两条路能够正确相加。

## 3. 数学解释

普通块直接拟合 `H(x)`，残差块写成 `y=F(x)+x`；若形状不同则 `y=F(x)+W_sx`。当所需变换接近恒等时，学习 `F≈0` 比重新拟合整个 `H` 更容易。对恒等捷径求导，`∂y/∂x=∂F/∂x+I`；梯度里有一项直接来自捷径。投影捷径 `W_s` 则保证 `[B,C,H,W]` 与主路输出形状匹配。本目录重点演示机制而非整个 ResNet 分层配置。

注意残差加法发生在特征图层面，不能把通道数为 8 的输入直接与通道数为 16 的主路输出相加。代码通过投影把两边都变成 [B,16,8,8]。如果人为把主路最后一层权重置零，块会主要由捷径传递输入；这就是“学习修正量”的直观边界情况。
