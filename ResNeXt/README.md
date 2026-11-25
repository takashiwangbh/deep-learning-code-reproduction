# ResNeXt：用更多并行变换扩大基数

## 1. 出处、提出者与意义

Saining Xie、Ross Girshick、Piotr Dollár、Zhuowen Tu 和 Kaiming He 在 2016 年 [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431) 中提出 ResNeXt。它将残差块内部拆成多条同构变换，再相加聚合；论文把分支数量称为 cardinality（基数），作为深度、宽度之外的模型容量维度。

## 2. 源码导航与直观解释

[resnext.py](resnext.py) 的 `ResNeXtBlock` 用 `groups=cardinality` 的 3×3 卷积高效表达多条并行分支，前后各用 1×1 卷积调整通道。`shortcut` 保留残差直通路径；`ResNeXt` 用几个这样的块做小图分类。输入 `[B,3,32,32]`，输出 `[B,类别数]`；运行 `python ResNeXt/resnext.py`。示例的层数与基数比论文标准 32×4d 版本小，重点是聚合机制。

## 3. 数学解释

一个块可抽象为 `y=x+Σ₍i=1..C₎Tᵢ(x)`，`C` 是基数。组卷积把中间通道分成 `C` 组，各组拥有独立的 3×3 卷积核，效果类似同时计算 `C` 条同拓扑变换。若中间通道为 `C·d`，每组宽度为 `d`；源码的 `middle=cardinality*group_width` 对应这个结构。形状变化时捷径用投影 `W_sx` 与主路对齐。

增大基数与单纯增大每组宽度不同：前者增加并行变换的组数，后者让每组更宽。源码把这两项分别暴露为 cardinality 和 group_width，便于在固定输入输出形状下比较参数量。注意 grouped 3×3 卷积的每一组只读取自己那一部分中间通道，前后的 1×1 卷积负责跨组混合。
