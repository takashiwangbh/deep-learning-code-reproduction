# Wide & Deep：记住组合，也泛化到新组合

## 1. 出处、提出者与意义

Heng-Tze Cheng、Levent Koc、Jeremiah Harmsen 等人在 2016 年的 [《Wide & Deep Learning for Recommender Systems》](https://arxiv.org/abs/1606.07792) 提出联合训练的推荐模型。Wide 线性部分能记住训练中出现过的特殊特征组合，例如某类用户和某类应用经常一起出现；Deep 部分把离散特征嵌入成连续向量，能对没见过的组合做一些泛化。把两部分相加训练，是这篇工作的核心。

## 2. 源码导航与直观解释

打开 [wide_and_deep.py](wide_and_deep.py)：`wide_dense` 处理连续特征，`wide_category` 处理离散 ID，`wide_cross` 处理预先构造的交叉特征 ID。Deep 分支把类别 ID 变成嵌入，再与连续特征拼接，经 MLP 输出一个分数。`forward` 将两路分数相加，返回 `[B]` 的点击 logit。`cross_ids` 由调用者预先把有意义的特征组合编码为 ID，源码没有假装自动发现业务交叉项。运行 `python 'Wide and Deep/wide_and_deep.py'` 看小批次反向传播。

## 3. 数学解释

Wide 分支可写成 `z_w = w_denseᵀx_dense + Σ_j w_cat[c_j] + Σ_k w_cross[g_k] + b`，其中 `g_k` 是显式交叉特征。Deep 分支 `z_d = MLP([x_dense, Σ_j E(c_j)])`，嵌入把离散 ID 映射到小维向量。最终概率 `P(click)=sigmoid(z_w+z_d)`，所以源码返回 `z_w+z_d` 并交给 `BCEWithLogitsLoss`。这里对多个类别向量求和，适合演示整体机制；复杂工业系统通常会保留字段位置、使用更丰富的交叉与特征工程。
