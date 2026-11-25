# ConvNeXt：现代化的纯卷积骨干

## 1. 出处、提出者与意义

Zhuang Liu 等人在 2022 年 [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) 中系统地更新传统 ResNet 的设计，形成 ConvNeXt。它继续使用卷积的局部性，同时借鉴当代模型的较大深度卷积核、层归一化、倒瓶颈式通道扩展和独立下采样。这个项目帮助比较 CNN 与视觉 Transformer 的结构选择。

## 2. 源码导航与直观解释

[convnext.py](convnext.py) 的 `ConvNeXtBlock` 用 7×7 depthwise 卷积收集空间邻域，转为 channels-last 后做 `LayerNorm`、两层逐位置线性变换和 GELU，再乘可学习的 `gamma` 加回输入。`ConvNeXt` 还展示 patchify 式 stem 与独立下采样。输入 `[B,3,32,32]`，输出 `[B,类别数]`；运行 `python ConvNeXt/convnext.py`。网络规模远小于论文标准模型，便于 CPU 阅读。

## 3. 数学解释

块内先算 `u=DWConv₇×₇(x)`；再对每个空间位置做 `v=W₂ GELU(W₁ LN(u))`，其中 `W₁` 将通道从 `C` 扩展到 `4C`，`W₂` 压回 `C`。最终 `y=x+γ⊙v`。深度卷积的参数量约为 `49C`，而同宽度普通 7×7 卷积约为 `49C²`。源码的 `gamma` 初值很小，使新块开始时接近恒等映射，有利于较深网络优化。

代码先把张量从 [B,C,H,W] 改排成 [B,H,W,C] 才交给 LayerNorm，因为该层默认规范化最后一维；计算完 MLP 再换回原布局。这个换轴只改变内存视图中的维度顺序，不会把像素与通道混成一维。若直接在 channels-first 张量上调用同一个 LayerNorm(C)，通常会因最后一维不是 C 而报错。
