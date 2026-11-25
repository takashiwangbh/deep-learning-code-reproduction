# Linformer：沿序列长度压缩键和值

## 1. 出处、提出者与意义

Sinong Wang、Belinda Z. Li、Madian Khabsa、Han Fang 和 Hao Ma 于 2020 年提出 [《Linformer: Self-Attention with Linear Complexity》](https://arxiv.org/abs/2006.04768)。普通注意力让 `n` 个 query 与 `n` 个 key 两两匹配，生成 `n×n` 权重。Linformer 利用注意力矩阵的近似低秩性质，把键和值在**序列长度维**投影到较短长度 `r`，再做匹配；当 `r` 固定时，主要注意力计算由 `n²` 降为 `nr`。

## 2. 源码导航与直观解释

在 [linformer.py](linformer.py) 的 `LinformerAttention` 中，普通线性层先产生 `Q,K,V: [B,H,n,d]`；可学习的 `E,F: [H,r,n]` 分别把 K 和 V 压成 `[B,H,r,d]`。随后每个原始 query 只与 `r` 个压缩后的 key 匹配，输出仍是每个原始位置一个向量。`Linformer` 给出最小 token 编码器，输入 `[B,n]`，输出 `[B,n,V]`。从根目录运行 `python Linformer/linformer.py`；改小 `rank` 可直接观察压缩程度。

本实现固定 `max_length` 并按实际长度裁切投影矩阵，聚焦编码器的非因果注意力。长度投影会混合前后词，直接拿来做自回归预测可能泄露未来；源码不声称实现因果 Linformer 或论文全部共享投影策略。

## 3. 数学解释

普通注意力为 `softmax(QKᵀ/√d)V`，其中 `QKᵀ` 大小为 `n×n`。Linformer 先计算 `K'=EK`、`V'=FV`，`E,F∈R^{r×n}`，于是变为 `softmax(QK'ᵀ/√d)V'`，权重仅有 `n×r`。源码里的两条 `einsum` 正对应 `EK` 和 `FV`，没有把通道 `d` 错误地当成压缩对象。若 `r` 过小，速度和内存更省，但不同位置的信息也可能被挤在一起；它是一种近似而非与完整注意力恒等。
