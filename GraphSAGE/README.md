# GraphSAGE：用邻居特征生成新节点表示

## 1. 出处、提出者与意义

William L. Hamilton、Rex Ying、Jure Leskovec 在 2017 年提出 [《Inductive Representation Learning on Large Graphs》](https://arxiv.org/abs/1706.02216)。早期图嵌入常给训练图中的每个节点存一个向量；遇到新节点时，没有现成的参数。GraphSAGE 改为学习一套**邻居采样与聚合函数**：只要新节点及其邻居有特征，就能生成表示，因此适合不断出现新用户、新论文的图。论文研究多种聚合器；这里实现最直观的均值聚合版本。

## 2. 源码导航与直观解释

[graphsage.py](graphsage.py) 中 `MeanSAGELayer.forward` 接收节点特征 `[N,F]` 和邻接矩阵 `[N,N]`。每一行先汇总该节点邻居的特征平均值，再与节点自己的特征拼接，经过共享线性层、ReLU 和 L2 归一化。`GraphSAGE` 堆两层后给每个节点输出类别 logits `[N,C]`。从仓库根目录运行 `python GraphSAGE/graphsage.py`，能看到小图节点分类的前向和反向传播。

这个文件用稠密邻接矩阵展示聚合算法，适合小图；大图训练时应按论文对邻居分层采样并使用稀疏存储。邻接矩阵只包含真正的邻居，自己节点由拼接的第一项单独保留；孤立节点的邻居平均值设为零。

## 3. 数学解释

第 `k` 层对节点 `v` 的邻居集合 `N(v)` 先算 `m_v^k = mean({h_u^{k-1}:u∈N(v)})`，再更新 `h_v^k = normalize(ReLU(W^k [h_v^{k-1};m_v^k]))`。分号表示把自己和邻居的向量接在一起；权重 `W^k` 对所有节点共享。源码里的 `adjacency @ x / degree` 就是一次性计算所有节点的邻居平均值。由于模型学的是共享函数而不是节点 ID 表，它可以用于有同类特征的新节点。两层聚合后，每个节点的表示能包含大约两跳范围内的信息。
