# RetinaNet：用 focal loss 处理密集背景

## 1. 出处、提出者与意义

Tsung-Yi Lin、Priya Goyal、Ross Girshick、Kaiming He、Piotr Dollár 于 2017 年提出 [《Focal Loss for Dense Object Detection》](https://arxiv.org/abs/1708.02002)，并设计 RetinaNet。单阶段检测器在大量位置和 anchor 上直接预测，容易遇到“多数都是简单背景”的类别失衡。Focal loss 给已经很容易分类的样本较小权重，让训练更关注难例。RetinaNet 同时用特征金字塔融合不同尺度的语义，再用共享的分类与回归子网做密集预测。

## 2. 源码导航与直观解释

[retinanet.py](retinanet.py) 中 `RetinaNet.forward` 先从两级主干构造自顶向下 FPN：深层 `p4` 上采样加到较浅的 `p3`。两个尺度复用同一 `cls_tower` 和 `box_tower`，每个网格位置生成三种比例 anchor。`focal_loss` 显式算概率、`p_t`、`(1−p_t)^γ` 与交叉熵。`training_loss` 用 IoU 做正、负、忽略匹配，正框回归偏移。输出类别 `[B,N,C]` 是各类独立 sigmoid logits，不含背景通道；框偏移 `[B,N,4]`。根目录运行 `python RetinaNet/retinanet.py`。

本目录仅用两级小 FPN 和较浅子网；原论文采用更多金字塔层、更多卷积和完整后处理。代码把分类末层偏置初始化为低前景先验，避免一开始所有背景都预测成高前景概率。

## 3. 数学解释

普通二元交叉熵是 `CE(p_t)=−log(p_t)`，其中 `p_t=p` 对正例、`1−p` 对负例。Focal loss 为 `FL=−α_t(1−p_t)^γ log(p_t)`：当例子已很容易、`p_t≈1`，因子 `(1−p_t)^γ` 很小；难例的因子较大。源码默认 `α=0.25,γ=2`。anchor 与真实框的 IoU `≥0.5` 为正、`<0.4` 为负，中间忽略；每个真实框的最佳 anchor 也设为正。位置回归用中心偏移与宽高对数偏移，并只在正框上计算 Smooth L1。分类损失除以正框数，使背景 anchor 总数变化时尺度更稳定。
