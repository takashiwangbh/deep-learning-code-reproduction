# GAT：让图节点自己决定关注哪些邻居

## 1. 出处、提出者与意义

Petar Veličković、Guillem Cucurull、Arantxa Casanova、Adriana Romero、Pietro Liò、Yoshua Bengio 在 2017 年提出 [《Graph Attention Networks》](https://arxiv.org/abs/1710.10903)。普通图卷积常根据图结构给邻居预定权重；GAT 让每条边的权重由两端节点当前特征学出来。例如论文引用图中，两个邻居都相连，但一个可能与中心论文的主题更相关。多头注意力让模型从不同特征角度判断邻居。

## 2. 源码导航与直观解释

[gat.py](gat.py) 的 `GraphAttentionLayer` 先以 `weight` 把每个节点投影到每个头的空间，再用 `attn_source`、`attn_target` 给节点对打分。`mask` 只让真实边及自身参与 softmax；没有边的位置不会泄露信息。第一层把多个头的输出拼接，第二层平均头并给出节点分类 logits `[N,C]`。根目录运行 `python GAT/gat.py` 可完成一张小图的前向和反向传播。为把公式展示得清楚，这里使用稠密 `[N,N]` 注意力矩阵，真实大图应使用边列表和稀疏计算。

## 3. 数学解释

令 `z_i^h=W^h h_i` 是第 `h` 个头给节点 `i` 的投影，边 `(i,j)` 的未归一化分数是 `e_ij^h = LeakyReLU((a_s^h)ᵀz_i^h + (a_t^h)ᵀz_j^h)`。只在 `j∈N(i)∪{i}` 上做 `α_ij^h = softmax_j(e_ij^h)`，然后 `h_i'^h=Σ_j α_ij^h z_j^h`。源码中 `masked_fill` 在 softmax 前屏蔽非邻居，`torch.matmul(weights, projected)` 是按权重聚合。拼接多个头提高表示宽度；最后分类层采用头平均，不让类别维数随头数增长。
