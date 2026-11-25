# MAE：只看少数图像块，重建被遮住的部分

## 1. 出处、提出者与意义

Kaiming He、Xinlei Chen、Saining Xie、Yanghao Li、Piotr Dollár、Ross Girshick 在 2021 年提出 [《Masked Autoencoders Are Scalable Vision Learners》](https://arxiv.org/abs/2111.06377)。MAE 把图像分块，随机遮掉很大一部分，只让编码器处理可见块，再由较小的解码器重建缺失像素。它的非对称设计节省了昂贵编码器的计算，也让没有人工标签的图像能够提供训练信号。论文常用约 75% 掩码比例，本目录也采用这一默认值。

## 2. 源码导航与直观解释

[mae.py](mae.py) 先用 `patchify` 把 `[B,3,32,32]` 转成 `[B,16,192]`：16 个 8×8 彩色块。`forward` 每张图独立打乱块序，只选可见索引送入 `encoder`；`decoder` 在原位置补入可学习的 `mask_token`，再预测所有块的像素，但损失**只统计遮住的位置**。它返回预测块、布尔掩码及标量 MSE。运行 `python MAE/mae.py` 看掩码数量和反向传播。本例固定图像大小、小型 Transformer，不提供预训练权重或论文下游分类结果。

## 3. 数学解释

设图像块为 `x₁,…,x_N`，随机选择可见集合 `V`，遮住集合 `M`。编码器只计算 `z=Encoder({Embed(x_i)+pos_i:i∈V})`，没有为遮住块浪费编码器计算。解码器把 `z` 放回原位置，并在 `M` 位置插入 mask token，预测 `x̂_i`。损失 `L=(1/|M|)Σ_{i∈M}‖x̂_i-normalize(x_i)‖²`，这里的 `normalize` 按每个块自己的像素均值与方差进行。源码用 `gather` 选可见块、`scatter_` 放回位置、`masked` 仅索引缺失块的误差；读这三处即可把公式对应到张量操作。
