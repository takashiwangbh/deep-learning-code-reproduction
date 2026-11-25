# Transformer-XL：让前一段成为下一段的记忆

## 1. 出处、提出者与意义

Zihang Dai、Zhilin Yang、Yiming Yang 等人在 2019 年提出 [《Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context》](https://arxiv.org/abs/1901.02860)。普通 Transformer 语言模型把长文本切成固定窗口时，下一窗口通常无法访问上一窗口的隐藏表示，依赖容易在边界处断开。Transformer-XL 缓存前一段各层的隐藏状态，并在下一段计算注意力时把它们作为额外的键和值；相对位置表示使跨段复用有意义。它是理解长上下文与状态缓存的基础。

## 2. 源码导航与直观解释

[transformer_xl.py](transformer_xl.py) 的 `RelativeAttention.forward` 把旧段 `memory` 接在当前段之前作为 K/V，但 Q 只来自当前段。`distance` 按“当前词距历史词多远”查相对位置向量；因果掩码阻止偷看当前段的未来。`TransformerXL.forward` 逐层接收旧记忆，返回本段 logits 与新的、已 `detach` 的记忆。输入本段 token `[B,T]`，记忆每层 `[B,M,D]`，输出 `[B,T,V]`。根目录运行 `python 'Transformer-XL/transformer_xl.py'` 可看到两段连续输入和反向传播。

教学实现用截断距离的可学习相对位置表，保留跨段相对关系；原论文使用更具体的正弦相对编码、位置变换及更完整的训练协议。这里不含大规模语料或预训练权重。

## 3. 数学解释

令当前隐藏状态为 `H_t`、上一段缓存为 `M_t`，键和值来自 `[SG(M_t); H_t]`，其中 `SG` 表示停止梯度：当前段能读取历史信息，但反向传播不跨越无限多段。源码在缓存时调用 `detach()`，并只保留最后 `mem_len` 个位置。注意力分数由内容项 `(q_i+u)·k_j` 和相对位置项 `(q_i+v)·r_{i-j}` 相加，再除以 `√d`、做因果 softmax；`u,v` 是每头的可学习偏置。与只使用绝对位置编号相比，`r_{i-j}` 让“相距三词”在不同段边界仍表示同一种距离。输出仍是正常的下一个 token 分数，训练目标为 `−log p(x_{t+1}|x_{≤t},M_t)`。
