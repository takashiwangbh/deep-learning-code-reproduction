# PVT：给密集视觉任务准备多尺度 Transformer 特征

## 1. 出处、提出者与意义

Wenhai Wang、Enze Xie、Xiang Li 等人于 2021 年提出 [《Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions》](https://arxiv.org/abs/2102.12122)。ViT 常把图像切成同一尺度的 patch 并用一个分类 token 汇总，而检测和分割需要不同分辨率的特征图。PVT 分阶段缩小 patch 网格、增加通道，生成类似 CNN 金字塔的多尺度表示；早期高分辨率阶段还把注意力的 K/V 在空间上缩小，以控制计算量。

## 2. 源码导航与直观解释

[pvt.py](pvt.py) 中 `SpatialReductionAttention` 保留完整分辨率的 query，却先用带步长的局部投影压缩 K/V，然后显式算 Q/K/V 注意力。`PVTBlock` 包含归一化、注意力、前馈与残差；`PVT.forward_features` 依次给出四级特征。示例输入 `[B,3,64,64]` 时，输出网格宽高依次为 16、8、4、2；每级缩减后的 K/V 仍有四个空间位置，可以观察注意力如何选择。分类头只拿最后一级做演示，检测或分割头可拿全部四级。根目录运行 `python PVT/pvt.py` 打印每级形状和 logits。

教学版每级一个块，patch 映射以卷积算子表达等价的局部线性投影；没有论文完整深度、位置编码设置或目标检测训练。重点是四级金字塔与 K/V 空间缩减在源码中如何对应。

## 3. 数学解释

阶段 `l` 的 patch 网格从 `H_l×W_l` 变到下一阶段约 `H_l/2×W_l/2`，通道宽度增加。若当前有 `N=H_lW_l` 个 query，空间缩减比为 `r`，则 K/V 长度约变成 `N/r²`。注意力从 `N×N` 变成 `N×(N/r²)`：`Q=XW_Q`，`K'=SR(X)W_K`，`V'=SR(X)W_V`，输出 `softmax(QK'ᵀ/√d)V'`。源码的 `sr` 对二维特征图做带步长投影，`k,v` 来自缩减后的 token；最终 `features` 保留每一级的二维位置，供密集预测复用。
