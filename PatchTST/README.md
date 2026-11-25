# PatchTST：把时间序列切成小段再预测

## 1. 出处、提出者与意义

Yuqi Nie、Nam H. Nguyen、Phanwadee Sinthong、Jayant Kalagnanam 在 2022 年的 [《A Time Series is Worth 64 Words》](https://arxiv.org/abs/2211.14730) 提出 PatchTST。直接把时间序列每一个时刻都当作 token，历史一长，自注意力的成本就很高。PatchTST 把相邻时间点组成一个 patch，注意力只在这些片段之间计算；同时采用**通道独立**设计，把每个变量单独作为一条序列，但所有变量共享同一套参数。这两点成为长期预测的重要基线。

## 2. 源码导航与直观解释

[patchtst.py](patchtst.py) 的 `forward` 接收 `[B,L,C]`：`B` 为批次，`L` 为历史长度，`C` 为变量数。`unfold` 以指定 `patch` 长度和 `stride` 步幅切片，再把 `B*C` 条单变量序列当作独立样本送入**同一个**线性嵌入和 Transformer 编码器。`forecast` 把每个变量的片段表示映射到未来 `H` 步，输出 `[B,H,C]`。根目录运行 `python PatchTST/patchtst.py`；可尝试只改一个输入通道，检查其他通道的预测是否保持不变。本例固定历史长度，只演示监督预测主干，未实现论文的实例归一化和自监督预训练流程。

## 3. 数学解释

给长度 `L` 的某个变量序列，以片长 `P`、步幅 `S` 切出 `N=⌊(L-P)/S⌋+1` 个片段。每个片段 `p_i∈R^P` 经过共享投影 `z_i=W p_i+b+position_i`，再由自注意力在 `N` 个片段之间建立关系。注意力矩阵从逐点建模的 `L×L` 缩为片段级的 `N×N`，所以长历史更易处理。源码把通道合并到 batch 维后调用同一编码器：模型参数不随 `C` 增加，也不存在跨通道注意力；最后按原变量维度重排。`Linear(N·D,H)` 为每个通道给出未来 H 步数值。
