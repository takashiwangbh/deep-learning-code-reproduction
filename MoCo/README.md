# MoCo：动量编码器与负样本队列

## 1. 出处、提出者与意义

Kaiming He、Haoqi Fan、Yuxin Wu、Saining Xie、Ross Girshick 在 2019 年提出 [《Momentum Contrast for Unsupervised Visual Representation Learning》](https://arxiv.org/abs/1911.05722)。自监督对比学习让同一图像的两种增强结果靠近、不同图像远离；若负样本只能来自当前小批次，可比较的样本数量有限。MoCo 维护一个跨批次的队列，并用缓慢变化的动量编码器生成队列中的键，使旧键与新键尽量保持一致。它让小批训练也能使用较大的负样本字典。

## 2. 源码导航与直观解释

[moco.py](moco.py) 的 `TinyImageEncoder` 是教学用卷积编码器，输出归一化的图像向量。`MoCo` 保存两套网络：查询编码器通过梯度训练；键编码器是它的移动平均副本，不接受梯度。`forward` 接收同一批图像的两种增强 `[B,3,H,W]`，把每组对应的查询/键作为正对，把队列里的旧键作为负例，算交叉熵后用新键替换队列最老位置。`queue_pointer` 显示环形队列写入位置。运行 `python MoCo/moco.py`。例子没有真实数据增强、批归一化打乱或 ImageNet 训练。

## 3. 数学解释

归一化查询 `q` 与对应键 `k⁺` 的相似度为 `q·k⁺`；队列中的键 `k_j⁻` 构成负例。对比损失 `L=-log[exp(q·k⁺/τ)/(exp(q·k⁺/τ)+Σ_j exp(q·k_j⁻/τ))]`，其中 `τ` 控制 softmax 温度。源码把正例放在 logits 第一列，所以类别标签恒为零。键编码器参数按 `θ_k ← m θ_k + (1-m)θ_q` 更新；`m` 接近 1 时变化缓慢。`queue` 是 buffer 而不是可训练参数，梯度只流向查询编码器。
