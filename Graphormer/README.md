# Graphormer：把图结构写进 Transformer 注意力

## 1. 出处、提出者与意义

Chengxuan Ying、Tianle Cai、Shengjie Luo、Shuxin Zheng、Guolin Ke、Di He、Yanming Shen、Tie-Yan Liu 在 2021 年提出 [《Do Transformers Really Perform Bad for Graph Representation?》](https://arxiv.org/abs/2106.05234)，模型名 Graphormer。普通 Transformer 只看一串 token，难以知道图中哪些节点相连、相隔几跳。Graphormer 给节点加入度数信息，并把最短路径距离和路径边信息放进注意力分数，使全局注意力具备图结构线索。它是图 Transformer 的代表性基线。

## 2. 源码导航与直观解释

[graphormer.py](graphormer.py) 的 `shortest_path_features` 用 BFS 为小图计算节点对的最短距离，以及一条最短路径上的边类型。`Graphormer.forward` 接收节点特征 `[B,N,F]`、邻接矩阵 `[B,N,N]` 和边类型矩阵 `[B,N,N]`；度数嵌入加入节点向量，距离/路径边嵌入加入注意力 logits。额外的 graph token 与所有节点交互，它的最终表示用于图分类 `[B,C]`。根目录运行 `python Graphormer/graphormer.py`。这里 BFS 和稠密注意力只适合教学小图；原论文还使用更丰富的中心性和路径边编码、深层网络与大规模训练。

## 3. 数学解释

对节点 `i`，输入向量为 `h_i=W x_i+e_degree(degree_i)`。注意力第 `h` 个头在节点对 `(i,j)` 上计算 `a_ij^h=(q_i^h·k_j^h)/√d+b_spatial^h(dist(i,j))+b_edge^h(path(i,j))`，再对所有 `j` 做 softmax。源码中的 `spatial_bias` 查最短路长度，`edge_bias` 对一条最短路径上的边类型嵌入取平均，之后加到点积注意力上。原论文的路径边编码有逐跳权重，这里用均值保留“沿路边类型影响注意力”的核心。Graph token 的结构偏置设为零，它仍可通过完整注意力聚合所有节点。改变图连接但不改变节点特征时，距离/度数及路径边偏置都会变化。
