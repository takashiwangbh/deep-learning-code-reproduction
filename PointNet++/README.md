# PointNet++：从局部点群逐层理解三维形状

## 1. 出处、提出者与意义

Charles R. Qi、Li Yi、Hao Su、Leonidas J. Guibas 在 2017 年提出 [《PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space》](https://arxiv.org/abs/1706.02413)。PointNet 用一个全局最大池化处理无序点集，但最开始每个点独立看自己，难以表达局部几何。PointNet++ 借鉴卷积网络的层次思路：选一批代表点、为每个代表点找附近点、在小邻域里用共享网络汇聚，再在更大的尺度重复。它是点云分类与分割的经典基础。

## 2. 源码导航与直观解释

[pointnet_plus_plus.py](pointnet_plus_plus.py) 的 `farthest_point_sample` 用最远点采样选出分散的中心；它从离点云质心最远的点起步，使无距离并列时结果不依赖输入排列。`radius_neighbors` 对每个中心找半径内的近邻，缺少邻居的槽位重复最近点。`SetAbstraction` 把邻居位置减去中心位置，与已有点特征拼接，经过共享 `1×1` 卷积并在邻居维取最大值。`PointNetPlusPlus` 用两级抽象把 `[B,N,3]` 变成 `[B,C]` 分类分数。运行 `python 'PointNet++/pointnet_plus_plus.py'`；CPU 例子只用 32 个点，没有论文的多尺度分组、密度自适应、特征传播分割头或大数据训练。

## 3. 数学解释

第 `l` 层先选中心集合 `S_l=FPS(X_l)`。对中心 `c`，半径查询给出邻域 `N_r(c)={x_i:||x_i-c||≤r}`；共享局部变换 `φ_l` 处理 `[(x_i-c);h_i]`，最后 `h'_c=max_{i∈N_r(c)} φ_l([(x_i-c);h_i])`。相对坐标让局部特征主要关心形状而非物体绝对位置；最大池化让邻居顺序不影响结果。第二层再对第一层中心及其特征重复，感受范围逐渐扩大。源码中 `torch.cdist`、`neighbor_ids`、`max(dim=-1)` 分别对应邻居选择、局部取值与对称汇聚。FPS 和邻居索引是离散操作，不对索引本身求梯度；局部网络参数仍可反向传播。
