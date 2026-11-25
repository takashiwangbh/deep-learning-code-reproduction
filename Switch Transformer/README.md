# Switch Transformer：每个 token 只选一个专家

## 1. 出处、提出者与意义

William Fedus、Barret Zoph、Noam Shazeer 于 2021 年提出 [《Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity》](https://arxiv.org/abs/2101.03961)。传统 Transformer 前馈层对所有 token 使用同一组权重；混合专家层为不同 token 提供多个前馈网络。Switch 的关键简化是**每个 token 只去概率最高的一个专家**，因此可增加总参数容量而不让每个 token 运行所有专家。路由负载与专家容量是读懂这类模型的关键。

## 2. 源码导航与直观解释

[switch_transformer.py](switch_transformer.py) 的 `SwitchFFN.router` 为 `[B,T,D]` 中每个 token 产生专家概率；`argmax` 选唯一专家，循环只运行该专家的 FFN。`capacity` 限制每个专家本批可接收的 token 数；超出容量者在这个子层输出零，但外层残差仍保留原表示。`auxiliary` 是辅助负载均衡项。`SwitchTransformer` 展示注意力层之后如何用专家 FFN 替换普通 FFN。输入 token ID `[B,T]`，返回 logits `[B,T,V]` 和辅助损失。根目录运行 `python 'Switch Transformer/switch_transformer.py'`。

教学版单机逐专家派发，未实现论文的大规模分布式通信、路由噪声、精确容量调度与 T5 完整编码器—解码器训练。这里的核心路由及超容量行为是可读源码。

## 3. 数学解释

路由器给 token `x` 计算 `p=softmax(W_rx)`，选择 `e=argmax_i p_i`，若专家 `e` 未满，则子层输出 `p_e·FFN_e(x)`；否则输出零，由外层 `x+...` 保留输入。容量约为 `ceil(capacity_factor·(B·T)/E)`，其中 `E` 为专家数。仅执行一个 FFN，计算成本更接近普通前馈层，而参数量可随 `E` 增长。辅助项 `L_aux=E·Σ_i f_i P_i`：`f_i` 是被选到专家 `i` 的 token 比例，`P_i` 是平均路由概率。源码返回该项，训练时需乘合适的小权重与主任务损失相加，缓解所有 token 挤向少数专家。
