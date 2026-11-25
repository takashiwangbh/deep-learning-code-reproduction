# DGCNN：在变化的特征空间重建点云邻居

## 1. 出处、提出者与意义

Yue Wang、Yongbin Sun、Ziwei Liu、Sanjay E. Sarma、Michael M. Bronstein、Justin M. Solomon 在 2018 年提出 [《Dynamic Graph CNN for Learning on Point Clouds》](https://arxiv.org/abs/1801.07829)。PointNet 独立处理每个点，再做全局池化；DGCNN 进一步让每个点看到附近点，并且每层重新选邻居。最初“附近”由三维坐标决定，后续则由学到的特征决定，因此原本空间上较远但语义相似的点也可能互相连接。关键模块叫 EdgeConv。

## 2. 源码导航与直观解释

[dgcnn.py](dgcnn.py) 中 `knn_indices` 计算每个点最近的 `k` 个邻居，排除自己。`EdgeConv.forward` 为每条边构造 `[中心特征, 邻居特征-中心特征]`，用共享的 `1×1` 卷积处理，再在邻居维取最大值。`DGCNN` 连续做两次 EdgeConv，第二次使用**第一层学出的特征**重新找邻居，最后做全局最大值、均值池化和分类。输入 `[B,N,3]`，输出 `[B,C]`；运行 `python DGCNN/dgcnn.py`。示例采用小点云和两层结构，以便 CPU 读懂；论文完整模型有更宽、更深的层与数据增强。

## 3. 数学解释

第 `l` 层先在当前特征 `h_i^l` 的距离空间构建 `N_k^l(i)`，然后 `e_ij^l = φ_l([h_i^l, h_j^l-h_i^l])`，`h_i^{l+1}=max_{j∈N_k^l(i)} e_ij^l`。`φ_l` 在源码中由共享的卷积与 ReLU 组成；`max` 让邻居顺序不影响结果。因为下一层的 kNN 使用新的 `h^{l+1}`，图会随网络层改变。选邻居是离散 top-k，索引本身不反向求导，但选出的边特征及其卷积参数可正常训练。源码用 `torch.cdist` 构建稠密距离矩阵，只适合教学小样本。
