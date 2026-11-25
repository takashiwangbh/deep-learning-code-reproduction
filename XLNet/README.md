# XLNet：随机预测顺序与双流注意力

## 1. 出处、提出者与意义

Zhilin Yang、Zihang Dai、Yiming Yang、Jaime Carbonell、Ruslan Salakhutdinov、Quoc V. Le 在 2019 年提出 [《XLNet: Generalized Autoregressive Pretraining for Language Understanding》](https://arxiv.org/abs/1906.08237)。普通自回归模型总从左到右预测，BERT 则遮盖输入位置。XLNet 改为对词位置的**预测顺序**随机排列：一个目标可以利用原句左侧或右侧、但在本次排列中更早的词；同时用内容流和查询流分开“读到哪些词”与“预测当前词”，防止把目标词直接看进去。这是理解置换语言建模的一项重要工作。

## 2. 源码导航与直观解释

[xlnet.py](xlnet.py) 的 `order` 每行列出本轮预测位置的排列，例如 `[1,0,2]` 表示先预测原句第 1 位，再预测第 0 位。`rank` 将它变成每个位置在排列中的先后名次。`TwoStreamLayer` 中内容流可读取自己的词和排列中更早的词；查询流只有位置和可学习查询 token，只能读取更早的内容，**不能读取自身词**。两流都可读取一个独立的起始记忆，保证第一个目标也有可用上下文。`training_loss` 对每个位置预测原词；运行 `python XLNet/xlnet.py`。本例完整展示置换掩码与双流防泄漏机制，但使用绝对位置嵌入，未实现原论文从 Transformer-XL 借来的相对位置编码、跨段记忆或部分预测策略。

## 3. 数学解释

设排列为 `z=(z₁,…,z_T)`，语言建模目标是 `E_z Σ_t log p_θ(x_{z_t}|x_{z_{<t}})`。对原位置 `i`、`j`，内容流可见条件是 `rank(j)≤rank(i)`；查询流可见条件严格为 `rank(j)<rank(i)`。源码里的 `content_allowed` 与 `query_allowed` 恰好相差一个等号。内容流 `h_i` 含真实词 `x_i`，所以不能直接拿它预测 `x_i`；查询流 `g_i` 的初始值只有位置，不含该词。每层的查询流只读**上一层**的内容流，从而在多层传播中也不会绕路看到目标。最后 `lm_head(g_i)` 预测 `x_i`，这就是双流注意力的必要性。
