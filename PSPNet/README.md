# PSPNet：用金字塔池化理解整张图的场景

## 1. 出处、提出者与意义

Hengshuang Zhao、Jianping Shi、Xiaojuan Qi、Xiaogang Wang、Jiaya Jia 在 2016 年提出 [《Pyramid Scene Parsing Network》](https://arxiv.org/abs/1612.01105)。语义分割要给每个像素一个类别，只看附近纹理可能把大物体的一小块认错。PSPNet 通过金字塔池化同时汇集整图、较大区域和较小区域的上下文，再把这些信息送回每个位置，使像素分类既能看局部又能理解全局场景。它是经典的上下文聚合分割基线。

## 2. 源码导航与直观解释

[pspnet.py](pspnet.py) 中 `PSPNet.encoder` 提取低分辨率特征，`PyramidPooling.forward` 分别把特征池化为 `1×1`、`2×2`、`3×3`、`6×6`，用 `1×1` 卷积处理后放大回原特征大小并拼接。`fuse` 混合这些尺度，分类头输出每类像素 logits，最后双线性上采样到原图 `[B,C,H,W]`。从根目录运行 `python PSPNet/pspnet.py`，可看到不规则尺寸 `48×40` 输入得到同尺寸分割输出。代码使用小卷积编码器，未包含论文的完整 ResNet 主干、辅助损失、多尺度评测和数据集训练。

## 3. 数学解释

若主干特征为 `X∈R^{B×D×h×w}`，每个尺度 `s∈{1,2,3,6}` 先做 `A_s=AdaptiveAvgPool_s(X)`，再得到 `U_s=Upsample(ReLU(Conv1×1(A_s)))`，使所有 `U_s` 回到 `h×w`。融合输入为 `[X;U_1;U_2;U_3;U_6]`，分号表示沿通道拼接；`Conv3×3` 混合后再预测像素类别。`s=1` 的池化让每个位置得到同一份全局概况，`s=6` 仍保留较细的区域线索。源码的 `adaptive_avg_pool2d`、`interpolate`、`torch.cat` 分别对应池化、对齐和融合。最终像素交叉熵可作用在上采样后的 logits 与分割标注上。
