# Seq2Seq：固定状态的编码器与解码器

## 1. 出处、提出者与意义

Ilya Sutskever、Oriol Vinyals、Quoc V. Le 在 2014 年 [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) 中用一个循环网络编码源序列，另一个循环网络自回归地产生目标序列，为神经机器翻译提供通用端到端框架。原论文使用深层 LSTM；本目录用手写 GRU 单元做较小演示，保留“编码成状态，再逐步解码”的关键机制。

## 2. 源码导航与直观解释

[seq2seq.py](seq2seq.py) 的 `GRUCell` 明确写出门控；`Seq2Seq.encode` 按顺序读取源 token，返回最终状态；`forward` 以该状态初始化解码器，用目标输入前缀（首位是 BOS）做 teacher forcing，输出 `[B,T,目标词表]` logits；`generate` 则把上一步预测当作下一步输入。源输入为 `[B,S]`。运行 `python Seq2Seq/seq2seq.py`。固定长度状态可能成为长句信息瓶颈，后续 Bahdanau Attention 项目对此作改进。

## 3. 数学解释

编码器反复计算 `h_s=f_enc(x_s,h_{s−1})`，最终 `c=h_S` 是源句摘要。解码器从 `s₀=c` 开始，计算 `s_t=f_dec(e(y_{t−1}),s_{t−1})`，再由 `softmax(Ws_t)` 给出下一个词分布。因此条件概率可分解为 `p(y₁:T|x)=∏_t p(y_t|y_<t,c)`。训练时提供真实前一词，推理时使用模型自己的前一预测；源码的 `forward` 与 `generate` 分别呈现两种路径。

训练样本的目标通常会拆成两段：解码输入为 [BOS, y1, ..., yT-1]，监督标签为 [y1, ..., yT]。如果把两者对齐成同一个 token，模型就会学会复制当前输入，而不是预测下一个词。generate 从 BOS 开始并逐步加入预测词，这与训练的 teacher forcing 形成清晰对照。
