# BART：通过修复被破坏的文本学习生成

## 1. 出处、提出者与意义

Mike Lewis、Yinhan Liu、Naman Goyal 等人在 2019 年提出 [《BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension》](https://arxiv.org/abs/1910.13461)。BART 故意破坏一段文本，再让模型恢复原文：编码器可以双向阅读已损坏文本，解码器则从左到右写出干净文本。这把 BERT 式理解与 GPT 式生成结合进一个编码器—解码器结构。论文比较多种噪声，其中“把连续片段换成一个 MASK”的 span infilling 是代表性做法。

## 2. 源码导航与直观解释

[bart.py](bart.py) 的 `span_infilling` 把原文中一整段连续 token 换成**一个**掩码 token，序列因此变短；示例保留开头 BOS 与末尾 EOS。`BART.forward` 先用双向编码器读取损坏文本，再让带因果掩码的解码器读取前面已写的 token，同时通过交叉注意力读取编码器记忆。`denoising_loss` 用干净序列右移一位作教师强制训练。输入、输出为 `[B,T]` token 与 `[B,T,V]` logits；根目录运行 `python BART/bart.py`。本例只实现单片段填空，不包含句子打乱、多噪声采样、预训练语料或论文大模型规模。

## 3. 数学解释

设噪声函数 `C(x)` 将原文一段 `[x_a,…,x_b]` 替换成单个 MASK。BART 最大化干净文本的条件概率 `p_θ(x|C(x))`，等价最小化 `L=-Σ_t log p_θ(x_t|x_{<t},C(x))`。编码器对 `C(x)` 的每个位置可以双向注意；解码器对自身位置 `t` 只看 `≤t`，以免看见要预测的未来词。源码的上三角 `causal` 掩码实现这个限制，`memory` 作为交叉注意力的 K/V。`clean[:, :-1]` 是解码输入，`clean[:, 1:]` 是目标，保证每一步实际预测下一个 token。
