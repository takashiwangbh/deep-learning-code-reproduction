# DiT：用 Transformer 预测扩散噪声

## 1. 出处、提出者与意义

William Peebles 与 Saining Xie 于 2022 年提出 [《Scalable Diffusion Models with Transformers》](https://arxiv.org/abs/2212.09748)。当时扩散模型常用 U-Net 去噪；DiT 说明可以把图像潜变量切成 patch，再交给 Transformer 处理，并研究模型宽度、深度与计算量如何影响生成质量。它连接了视觉 Transformer 与扩散生成，是理解近年图像生成骨干的一条清晰路线。此目录只实现时间条件噪声预测网络，可与 DDPM 配合；不包括大规模预训练 VAE 或论文训练配置。

## 2. 源码导航与直观解释

源码：[dit.py](dit.py)。`DiT.patch_embed` 用步长等于 patch 大小的卷积把 `[B,C,H,W]` 变成 patch 序列；`position_embedding_2d` 标记每块位置；`timestep_embedding` 告诉网络现在噪声有多重。各 `DiTBlock` 让 patch 互相交流，并通过 `adaLN` 的 shift、scale、gate 接受时间条件；`FinalLayer` 再把每个 token 还原为一个 patch，拼回 `[B,C,H,W]` 的预测噪声。可以想成“把画面分成小方块，让方块交换信息后分别回答本处噪声”。从仓库根目录运行 `python DiT/dit.py`，小尺寸示例会检查输出和梯度。

## 3. 数学解释

若边长为 `H`、patch 边长为 `P`，序列长度就是 `(H/P)²`；源码 `Conv2d(kernel_size=P,stride=P)` 实现切块兼投影。每个块的注意力用 `softmax(QKᵀ/√d)V` 混合不同位置的信息。时间条件向量 `c` 在 `modulate` 中改变归一化特征：`h'=(1+scale(c))⊙LN(h)+shift(c)`；残差输出再乘 `gate(c)`，使去噪能力随时间步变化。`DiT.__init__` 将条件门和最终输出层初始化为零，起始时各残差分支近似恒等，预测噪声为零，之后再学习有效更新。训练时可将 `DiT(x_t,t)` 交给 DDPM 的均方误差目标 `‖ε−ε_θ(x_t,t)‖²`；本文件专注于 `ε_θ` 的源码。
