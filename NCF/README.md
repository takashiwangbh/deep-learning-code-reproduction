# NCF：神经协同过滤

## 1. 出处、提出者与意义

Xiangnan He、Lizi Liao、Hanwang Zhang、Liqiang Nie、Xia Hu、Tat-Seng Chua 于 2017 年提出 [《Neural Collaborative Filtering》](https://arxiv.org/abs/1708.05031)。传统矩阵分解用用户向量与物品向量的内积预测偏好，表达方式比较固定。NCF 用神经网络学习用户与物品的非线性交互；论文中的 NeuMF 把逐元素相乘的 GMF 分支与多层感知机分支结合，兼顾简单的协同信号和更灵活的关系建模。

## 2. 源码导航与直观解释

[ncf.py](ncf.py) 中 `NeuMF` 有两套独立嵌入：`gmf_user/item` 先逐元素相乘，`mlp_user/item` 则拼接后送进 MLP。最后把两条路得到的特征拼接，`output` 给出每对用户和物品的一个点击 logit，形状 `[B]`。运行 `python NCF/ncf.py` 可看到正负样本的二元交叉熵和反向传播。真实推荐训练还需从未交互物品中构造负样本，并在时间或用户维度划分评估集；随机例子只验证源码。

## 3. 数学解释

设用户和物品的 GMF 向量为 `pᵤ,qᵢ`，则 `φ_GMF=pᵤ⊙qᵢ`；另两组嵌入经拼接和逐层变换得到 `φ_MLP=MLP([p'ᵤ,q'ᵢ])`。NeuMF 输出 `logit=wᵀ[φ_GMF,φ_MLP]+b`，点击概率为 `sigmoid(logit)`。源码直接返回 logit，配合 `BCEWithLogitsLoss`，它在数值上比先 sigmoid 再算对数损失更稳定。一个 batch 的 user/item ID 都是 `[B]`，两个分支的参数各自学习，因此不会被迫共享同一种交互表达。
