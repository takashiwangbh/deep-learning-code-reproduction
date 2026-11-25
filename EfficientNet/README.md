# EfficientNet：平衡宽度、深度与分辨率

## 1. 出处、提出者与意义

Mingxing Tan 和 Quoc V. Le 在 2019 年 [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) 中指出，算力增加时只加深或只加宽网络并非最佳选择；把深度、宽度和输入分辨率按共同系数协调增长，通常更有效。EfficientNet 的骨干块采用 MBConv，并结合 SE 与 SiLU 激活。

## 2. 源码导航与直观解释

[efficientnet.py](efficientnet.py) 的 `MBConv` 依次扩展通道、做深度卷积、用 SE 门调整通道，再投影回输出通道；形状不变时使用残差。`EfficientNet` 用 `width_scale` 调通道数、`depth_scale` 调重复次数；输入分辨率由传入图片控制。运行 `python EfficientNet/efficientnet.py`，检查 `[B,3,32,32]→[B,类别数]`。例子并非原论文搜索得到的 B0 全量配置，也没有训练日程。

## 3. 数学解释

复合缩放可写成深度 `d=α^φ`、宽度 `w=β^φ`、分辨率 `r=γ^φ`，约束 `αβ²γ²≈2`，其中 `φ` 控制新增算力。源码展示宽度与深度两个可直接编码的维度，分辨率通过输入大小选择。MBConv 内部先从 `C` 扩展到 `eC`，深度卷积仅在各通道内提取空间特征，SE 产生每通道权重，最后 1×1 投影到 `C_out`；当 `C=C_out` 且步长为 1 时加回输入。

例如 width_scale 增大时，scaled_channels 会把每阶段通道数按硬件友好的倍数取整；depth_scale 增大时，math.ceil 会提高每阶段 MBConv 的重复次数。二者只改变骨干容量，不会自动改变数据集与训练策略。原论文还同时调整输入分辨率，本教学代码让调用者自行选择图像大小，避免误称已经完整实现复合缩放搜索。
