# Highway Network：用门控制信息是变换还是直通

## 1. 出处、提出者与意义

Rupesh Kumar Srivastava、Klaus Greff、Jürgen Schmidhuber 于 2015 年提出 [《Highway Networks》](https://arxiv.org/abs/1505.00387)。网络层数增加时，训练梯度更难穿过很多非线性变换。Highway 层为每个特征学一个门：需要新表示时让变换通过，需要保留旧表示时让原输入直接通过。它在残差网络广泛使用之前，展示了深层网络中显式信息通路的价值。

## 2. 源码导航与直观解释

[highway.py](highway.py) 的 `HighwayLayer` 有两套线性层：`transform` 生成新特征，`gate` 经 sigmoid 生成 `[B,D]` 的 0 到 1 选择权重；`forward` 把新旧特征按门加权相加。`HighwayNetwork` 先把输入映射到固定宽度，再连续堆叠多个 Highway 层并分类。运行 `python 'Highway Network/highway.py'` 可看到八层小模型的 `[B,C]` 输出及梯度。门偏置初始化为负数，开头更偏向让输入直通，便于深层网络起步训练。这里用全连接层演示门控，不包含论文完整数据集训练。

## 3. 数学解释

写 `H(x)=ReLU(W_Hx+b_H)`、`T(x)=sigmoid(W_Tx+b_T)`，则 Highway 层 `y=H(x)⊙T(x)+x⊙(1-T(x))`，`⊙` 是逐元素乘法。若某维 `T≈0`，该维几乎原样通过；若 `T≈1`，该维主要使用新变换。代码中 `gate * transformed + (1-gate) * x` 与公式逐项对应。相比固定系数为 1 的残差加法，Highway 的两条路由一个可学习门逐特征分配，代价是额外的门参数。初始化 `b_T=-2` 使 `sigmoid(-2)≈0.12`，初始时更多输入能沿直通路径传递。
