# GoogLeNet：同一层并行看多种尺度

## 1. 出处、提出者与意义

Christian Szegedy、Wei Liu、Yangqing Jia 等研究者在 2014 年发表 [《Going Deeper with Convolutions》](https://arxiv.org/abs/1409.4842)，提出 Inception 架构；GoogLeNet 是其中的一个 22 层实例。普通网络每层只能选一种卷积大小，Inception 让 1×1、3×3、5×5 和池化分支并行，再汇合结果。它解决的实际问题是：在计算预算有限时，怎样同时扩大网络深度、宽度和特征尺度，而不让运算量失控。

## 2. 源码导航与直观解释

先看 [googlenet.py](googlenet.py) 中的 `Inception`：`branch1`、`branch3`、`branch5`、`branch_pool` 分别走不同路线，`forward` 用 `torch.cat(..., dim=1)` 把它们的结果按通道拼起来。再看 `GoogLeNet` 如何堆叠九个这样的模块。`AuxiliaryClassifier` 是训练时从网络中段引出的辅助分类头，给前层提供更短的梯度路径。训练且 `aux_logits=True` 时输出 `(主 logits, 辅助1, 辅助2)`；评估时只输出主 logits，形状均为 `[B,类别数]`。

从仓库根目录运行 `python "GoogLeNet/googlenet.py"`。脚本底部使用小通道并关闭辅助头以便快速运行。教学版辅助头用自适应池化兼容小图片；原论文大图尺寸和训练配方需要另行准备。

## 3. 数学解释

假设输入有 `C_in` 个通道，直接做 3×3 卷积到 `C_out` 约需 `9C_inC_out` 个权重。若先用 1×1 卷积降到 `C_mid`，再做 3×3，权重约为 `C_inC_mid+9C_midC_out`；当 `C_mid` 较小时更省。源码 `branch3`/`branch5` 中先降维的 `_conv(...,1)` 就在做这件事。四个分支的输出宽高相同，故可拼接为 `C_total=C_1+C_3+C_5+C_pool` 个通道；`Inception.out_channels` 正是这个和。并行分支像是同时用小、中、大窗口观察图像，后续层再决定哪些线索有用。
