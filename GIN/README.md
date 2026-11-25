# GIN：保留邻居个数的图同构网络

## 1. 出处、提出者与意义

Keyulu Xu、Weihua Hu、Jure Leskovec、Stefanie Jegelka 在 2018 年的 [《How Powerful are Graph Neural Networks?》](https://arxiv.org/abs/1810.00826) 分析了图网络区分结构的能力，并提出 Graph Isomorphism Network（GIN）。对邻居求平均可能把“一个特征为 A 的邻居”和“两个特征为 A 的邻居”混在一起；GIN 使用求和，保留数量信息，再通过 MLP 学出合适的变换。它是学习图结构表达能力的经典模型。

## 2. 源码导航与直观解释

[gin.py](gin.py) 中 `GINLayer` 把所有邻居特征直接求和，再加上带系数的自身特征，送入两层 MLP。`epsilon` 是可学习的标量，控制自身信息的权重。`GINClassifier` 每层都把所有节点特征求和成一个图表示，最后把不同层的图表示拼接并分类；输入为单张图的 `[N,F]` 节点特征与 `[N,N]` 邻接矩阵，输出 `[C]` 图类别 logits。运行 `python GIN/gin.py`。这里用稠密小图说明机制，未包含大批量图的分段池化。

## 3. 数学解释

对节点 `v`，一层更新是 `h_v^k=MLP_k((1+ε_k)h_v^{k-1}+Σ_{u∈N(v)}h_u^{k-1})`。与平均相比，求和能区分同一特征出现一次还是多次；`ε_k` 让模型选择自身特征相对邻居的权重。源码中 `adjacency @ x` 实现邻居求和，注意邻接矩阵不含对角线，否则自身会被数两次。图级读出 `g_k=Σ_v h_v^k`，再把 `g_0,…,g_K` 拼接分类；保留浅层和深层读出，模型既能利用局部原始特征，也能利用多跳结构。
