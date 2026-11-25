# Bahdanau Attention：解码时重新查找源句

## 1. 出处、提出者与意义

Dzmitry Bahdanau、Kyunghyun Cho、Yoshua Bengio 在 2014 年 [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) 中指出，把整句压成一个固定向量会限制编码器—解码器。他们让解码器每生成一个词，就对源句所有位置计算软对齐，动态读取当前最相关的信息。这里实现这种加性注意力，而不只调用现成注意力层。

## 2. 源码导航与直观解释

[bahdanau_attention.py](bahdanau_attention.py) 的 `AdditiveAttention.forward` 使用解码隐藏状态作 query，所有编码位置的状态作 values，计算对齐权重与加权上下文。`BahdanauSeq2Seq` 本地写出 GRU 编码、注意力读取、GRU 解码与输出词分数。源 `[B,S]`，目标输入 `[B,T]`，返回 logits `[B,T,V]` 和对齐矩阵 `[B,T,S]`；运行 `python 'Bahdanau Attention/bahdanau_attention.py'`。传入 `source_valid` 时，编码器跳过 PAD 位置的状态更新，注意力也不会读取这些位置，因此改动 PAD token ID 不会影响有效输出。原论文用双向编码器，本教学版为便于逐行理解采用单向编码器。

## 3. 数学解释

对于解码前一状态 `s_{t−1}` 和源位置状态 `h_j`，匹配分数 `e_{tj}=vᵀtanh(W_s s_{t−1}+W_h h_j)`。归一化后 `a_{tj}=softmax_j(e_{tj})`，上下文 `c_t=Σ_j a_{tj}h_j`。源码的 `query/key/score` 三个线性层和 `torch.bmm` 分别实现打分与加权读取。解码器再用 `[目标词嵌入;c_t]` 更新状态。若源句有 padding，`source_valid` 会在 softmax 前屏蔽无效位置。
