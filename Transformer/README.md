# Transformer：编码器与解码器

## 1. 出处、提出者与意义

出处是 Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N. Gomez、Lukasz Kaiser 与 Illia Polosukhin 2017 年的 [《Attention Is All You Need》](https://arxiv.org/abs/1706.03762)。此前的机器翻译系统通常逐词运行循环网络。Transformer 用注意力在一个层内直接连接不同位置，训练时可以并行处理一句话中的所有词；论文同时给出了完整的编码器—解码器结构。

它成为后续 BERT、GPT 与视觉 Transformer 的基础。这里复现的是原始的**双塔翻译式结构**：编码器读源句，解码器既看已经生成的目标词，也看编码器输出。代码只呈现网络和掩码，不包含原论文的数据、分词、学习率日程或训练结果。

## 2. 源码导航与直观解释

打开 [transformer.py](transformer.py)，建议依次读 `MultiHeadAttention`、`PositionalEncoding`、`EncoderLayer`、`DecoderLayer`、`TransformerSeq2Seq.forward`。输入 `source_ids` 为 `[B,S]`，目标前缀 `target_input_ids` 为 `[B,T]`，输出下一词 logits 为 `[B,T,V目标]`。`EncoderLayer` 的自注意力让源句各词互相读取；`DecoderLayer` 的第一个注意力只能读到当前及此前目标词，第二个注意力从编码器读取源句。

直观地看，编码器先给源句每个词做一张“含上下文的卡片”；解码器每写一个目标词时，一边查看已写的词，一边检索这些卡片。`source_mask` 排除 PAD，`target_mask` 还要阻止偷看未来目标词。仓库根目录运行 `python Transformer/transformer.py`，只用随机 token 检查形状。阅读时可用 [序列检查](../checks/test_sequence.py) 中的因果性例子验证：改动未来词不应改变过去位置的输出。

## 3. 数学解释

给每个词的表示做三个线性变换，得到查询 `Q`、键 `K`、值 `V`。注意力计算 `Attention(Q,K,V) = softmax(QKᵀ/√dₖ) V`：`QKᵀ` 是“这个位置想找的信息”与“每个位置提供的信息”的匹配分数；除以 `√dₖ` 防止维度大时分数过大；softmax 把分数变为权重，最后加权读取 `V`。`MultiHeadAttention.forward` 中的矩阵乘法与此式逐项对应。多头把宽度拆成数个较小的子空间，各头各自读取后拼接，能同时关注不同关系。

仅有注意力并不知道词序，因此 `PositionalEncoding` 给偶数维加入 `sin(pos/10000^(2i/D))`、奇数维加入 `cos(pos/10000^(2i/D))`。位置 `pos` 与维度 `i` 决定不同频率；模型由这些变化区分词的顺序。每个层再用残差连接与层归一化保存输入信息，前馈网络逐位置变换特征。

解码器第 `t` 行注意力只允许列 `≤t`：`causal_mask` 生成下三角布尔矩阵，未来位置在 softmax 前被遮蔽。交叉注意力则用解码状态作 Q、编码器状态作 K/V，所以目标句每个位置可读取整个有效源句。训练标签相对输入右移一位，预测结果才是真正的“下一个词”。
