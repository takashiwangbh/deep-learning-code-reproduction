# MLP-Mixer：不使用卷积或注意力的图像分类

## 1. 出处、提出者与意义

Ilya Tolstikhin、Neil Houlsby、Alexander Kolesnikov 等人在 2021 年提出 [《MLP-Mixer: An all-MLP Architecture for Vision》](https://arxiv.org/abs/2105.01601)。它把图片切成 patch 后，只用两种多层感知机：一种跨 patch 混合位置信息，一种在每个 patch 内混合通道。这个简洁结构说明，在足够的数据和训练配方下，图像分类并不一定要求卷积或注意力；它也使“空间混合”和“通道混合”两个方向容易分开理解。

## 2. 源码导航与直观解释

[mlp_mixer.py](mlp_mixer.py) 的 `MLPMixer.patches` 把图像变成 `[B,N,C]` patch 序列。`MixerBlock.token_mlp` 先转置成 `[B,C,N]`，让同一通道的不同 patch 通过两层线性映射交流；转回来后，`channel_mlp` 对每个 patch 的 `C` 个通道做变换。两步各有 LayerNorm 和残差。最后对所有 patch 求平均并分类。根目录运行 `python 'MLP-Mixer/mlp_mixer.py'`，默认输入 32×32 图像，输出 `[B,10]`。

本项目使用小尺寸以便 CPU 演示；没有论文的大规模预训练、正则化和训练精度。由于 token 混合层直接依赖 patch 数，若改变图像尺寸，初始化时也要重新设置 `image_size` 或重建模型。

## 3. 数学解释

设 patch 表示 `X∈R^{N×C}`，`N` 是 patch 数，`C` 是通道宽度。第一条残差可写为 `U=X+[W₂σ(W₁ LN(X)ᵀ)]ᵀ`：转置后线性层沿 `N` 轴工作，所有通道共享同一套 token 混合权重。第二条为 `Y=U+W₄σ(W₃ LN(U))`：线性层沿 `C` 轴工作，各 patch 共享权重。源码中两次 `x + ...` 对应这两条残差。与自注意力不同，token 混合矩阵不随本张图的内容重新计算；模型靠训练学一组固定的空间交互权重。
