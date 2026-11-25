# DINO：视觉 Transformer 的无标签自蒸馏

## 1. 出处、提出者与意义

Mathilde Caron、Hugo Touvron、Ishan Misra、Hervé Jégou、Julien Mairal、Piotr Bojanowski、Armand Joulin 在 2021 年提出 [《Emerging Properties in Self-Supervised Vision Transformers》](https://arxiv.org/abs/2104.14294)。DINO 让学生网络预测教师网络对另一种图像视图给出的概率分布，教师又是学生的移动平均版本，不需要类别标签。把这种自蒸馏用在 ViT 上，可学到对图像内容有意义的特征与注意力。论文的多视图裁剪、教师中心化和锐化共同帮助训练稳定。

## 2. 源码导航与直观解释

[dino.py](dino.py) 里的 `TinyVisionTransformer` 先用卷积切图像块，加入位置和 CLS token，再通过 Transformer 编码器和投影头得到 `[B,K]` 的原型 logits。`DINO.forward` 对两种视图都求学生与教师输出；学生 A 拟合教师 B，学生 B 拟合教师 A。教师目标用温度和 `center` 调整，`update_teacher` 在学生优化器步骤后执行 EMA。运行 `python DINO/dino.py` 看 CPU 小模型的一次更新。为了清楚展示核心，这里只用两种同尺寸视图；完整论文有多裁剪视图、优化日程及大规模训练。

## 3. 数学解释

教师概率 `p_t = softmax((g_t(x)-c)/τ_t)`，其中中心向量 `c` 减去各原型的平均偏置，较低的 `τ_t` 让分布更尖锐。学生概率 `p_s=softmax(g_s(x')/τ_s)`，跨视图损失为 `-Σ_k p_t^k log p_s^k`；方向互换后取平均。源码中的 `prob_a/prob_b` 与 `log_a/log_b` 正好对应两种分布。中心以 batch 教师输出的移动平均更新；教师权重按 `θ_t ← m θ_t + (1-m)θ_s` 更新，且没有梯度。这样教师给出较稳的训练目标，学生从不同裁剪图中学习保持相关的语义。
