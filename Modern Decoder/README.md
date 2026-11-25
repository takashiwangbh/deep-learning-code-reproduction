# Modern Decoder（现代自回归解码器）

## 1. 出处、提出者与意义

这个项目展示**几项论文机制如何组合**，并不把它冒称为某个发布模型的完整复现。旋转位置编码 RoPE 来自 Jianlin Su 等人的 [RoFormer](https://arxiv.org/abs/2104.09864)（2021）；分组查询注意力 GQA 来自 Joshua Ainslie 等人的 [GQA](https://arxiv.org/abs/2305.13245)（2023）；RMSNorm 来自 Biao Zhang 与 Rico Sennrich 的 [论文](https://arxiv.org/abs/1910.07467)（2019）；SwiGLU 的对比研究来自 Noam Shazeer 的 [GLU Variants](https://arxiv.org/abs/2002.05202)（2020）。稀疏专家的核心思想可追溯到 Noam Shazeer 等人的 [Sparsely-Gated Mixture-of-Experts](https://arxiv.org/abs/1701.06538)（2017）。

这些机制分别解决位置表达、注意力缓存、归一化、前馈变换和有条件增加模型容量的问题。先读 GPT 项目里的普通因果注意力，再看本项目，会更容易发现每一步增加了什么。这里没有大规模训练数据、权重或发布模型的训练系统。

## 2. 源码导航与直观解释

所有核心代码都在 [modern_decoder.py](modern_decoder.py)。按 `RMSNorm → apply_rope → GroupedQueryAttention → SwiGLU → SparseMoE → DecoderBlock → ModernDecoderLM` 的顺序阅读。输入 token ID 为 `[B,T]`；嵌入后为 `[B,T,D]`；输出 `logits` 为 `[B,T,V]`，分别表示每个位置对词表中 `V` 个 token 的分数。`ModernDecoderLM.forward` 还返回 MoE 辅助损失，可与语言模型交叉熵一起训练。

`GroupedQueryAttention` 在分数矩阵中遮住未来位置，因此当前位置不能偷看后面的 token。`SparseMoE` 给每个 token 选择少数几个专家，只计算选中专家的前馈网络。项目底部用随机 token 做一次前向、交叉熵和反向传播；从仓库根目录运行 `python 'Modern Decoder/modern_decoder.py'`。若只想看基础结构，可把 `num_experts=0`，前馈部分就换成普通 SwiGLU。

## 3. 数学解释

注意力先投影出 `Q,K,V`，再计算 `softmax(QKᵀ/√d_head + mask)V`。`mask` 把未来 token 的分数设为极小值；softmax 后它们的权重接近 0。RoPE 在点积前旋转 Q 和 K 的每对维度，旋转角度取决于 token 位置；这样相对位置会影响 `QKᵀ`。GQA 则让 `H` 组 Q 共享较少的 `K` 组 K/V（`K≤H` 且 `H` 能被 `K` 整除），所以 K/V 参数和推理缓存都更小；源码中的 `repeat_interleave` 展示了共享关系。

RMSNorm 计算 `x / √(mean(x²)+ε) · γ`，其中 `γ` 可学习；它与 LayerNorm 的区别是没有减均值。SwiGLU 计算 `W_down[SiLU(W_gate x) ⊙ W_up x]`，门控支路决定哪些特征继续通过。MoE 先用路由器得到各专家分数，选出 top-k，再按归一化的门控权重对专家输出求和。`SparseMoE` 还返回负载均衡项，减少所有 token 都挤向同一专家的倾向；这份易读实现没有分布式专家调度或容量限制。
