# CLIP：让图像和文字进入同一特征空间

## 1. 出处、提出者与意义

Alec Radford、Jong Wook Kim、Chris Hallacy 等作者于 2021 年提出 [《Learning Transferable Visual Models From Natural Language Supervision》](https://arxiv.org/abs/2103.00020)。传统图像分类常只能识别预先固定的一组标签；CLIP 利用成对的图像和文字，让模型学习“这段话与哪张图相配”。训练后，任务可用自然语言描述候选类别，再比较图文相似度，不必为每个新类别重写分类头。这使图文对齐和零样本迁移成为重要研究方向。此项目保留双编码器与对称对比损失，未附带原论文的大规模图文数据、分词器和预训练权重。

## 2. 源码导航与直观解释

源码：[clip.py](clip.py)。`ImageEncoder` 把图像切成 patch，用 Transformer 得到图像向量；`TextEncoder` 将 token 序列编码，取最后一个有效 token（通常是 EOT）的向量。`CLIP.forward` 把两类向量归一化后，输出 `[B,B]` 的相似度矩阵：行是图片，列是文本；同下标的一图一文应配对。`contrastive_loss` 同时从“给图找文”和“给文找图”两个方向训练。可以把它想成一场配对游戏：每个 batch 的正确答案都在矩阵对角线上。根目录运行 `python CLIP/clip.py`，小型随机例子检查损失和梯度。

## 3. 数学解释

令图像编码为 `u_i=f(image_i)`、文字编码为 `v_j=g(text_j)`；先做 `û_i=u_i/‖u_i‖`、`v̂_j=v_j/‖v_j‖`，再计算 `s_{ij}=exp(τ)·û_iᵀv̂_j`。归一化后的点积就是余弦相似度，`τ` 是可学习温度参数；源码 `CLIP.forward` 显式构造所有 `i,j` 的 `s`。对第 `i` 张图，图到文损失为 `−log(exp(s_{ii})/Σ_j exp(s_{ij}))`；文到图交换行列再算一次，`contrastive_loss` 取二者平均。因此正确图文对的分数会被推高，错误组合被推低。批内需要至少两对样本才能形成负例；如果配对顺序弄错，对角线目标也会错。
