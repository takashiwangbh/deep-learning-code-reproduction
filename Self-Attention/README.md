# Self-Attention：缩放点积与多头读取

## 1. 出处、提出者与意义

Ashish Vaswani 等人在 2017 年 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 中以多头自注意力为核心构建 Transformer。序列中每个位置可以根据内容直接读取其它位置，而不必等待循环网络逐步传递；多头机制让不同子空间同时关注不同关系。本项目抽出这一机制，方便在学习完整 Transformer 前逐行理解。

## 2. 源码导航与直观解释

[self_attention.py](self_attention.py) 中的 `scaled_dot_product` 明确计算分数、掩码、softmax 和加权求和；`MultiHeadSelfAttention` 则把宽度 `D` 分成 `H` 个头，投影出 `Q/K/V` 并合并各头。输入 `[B,T,D]`，输出 `[B,T,D]`，注意力权重 `[B,H,T,T]`。运行 `python 'Self-Attention/self_attention.py'`；把 `causal=True` 打开时，上三角未来位置被屏蔽。

## 3. 数学解释

单头注意力为 `softmax(QKᵀ/√dₖ)V`。`QKᵀ` 的第 `(i,j)` 项是位置 `i` 对位置 `j` 的匹配分数；除以 `√dₖ` 防止维度增加时分数过大。softmax 将每行转为读取权重，再加权求和 `V`。多头把 `D` 均分为 `H` 个 `dₖ=D/H` 子空间，分别计算后拼接并线性投影。因果掩码保证位置 `i` 只能看到 `j≤i`，因此可用于自回归生成。

设输入长度为 5，则注意力权重矩阵为 5×5；在因果模式下，第 0 行只能读第 0 列，第 4 行可读全部前五列。把第 4 个 token 改掉，前四个位置的输出应该保持不变。检查脚本同时验证上三角权重为零和这一前缀不变性，避免仅仅看输出形状就误以为掩码正确。
