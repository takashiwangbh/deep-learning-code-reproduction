# Factorization Machines：稀疏特征的成对交互

## 1. 出处、提出者与意义

Steffen Rendle 于 2010 年在 [《Factorization Machines》](https://doi.org/10.1109/ICDM.2010.127) 中提出因子分解机（FM）。推荐和点击率数据通常很稀疏：一个用户、一个物品及少数上下文特征会被激活，大多数组合很少出现。普通线性模型只看单个特征；若给每对特征单独设参数，又会因组合太多而很难学到。FM 为每个特征学一个短向量，用两向量的内积估计成对作用，即使某对特征出现次数少，也能借它们在其他组合中学到的表示进行预测。后来的 DeepFM 沿用了这一低阶交互部分。

## 2. 源码导航与直观解释

[factorization_machines.py](factorization_machines.py) 中 `FactorizationMachine` 接收 `[B,F]` 的实值特征矩阵，输出 `[B]` 的分数。`bias` 是整体偏置，`linear` 给每个特征一个一阶权重，`factors` 为每个特征存一个低维向量。`forward` 把线性分数与所有不同特征对的交互分数相加；交互项用平方和恒等式一次算出，不建立 `F×F` 参数表。仓库根目录运行 `python 'Factorization Machines/factorization_machines.py'`，用两行稀疏输入演示二元交叉熵及反向传播。

本地实现使用稠密张量来清楚展示公式。实际超大规模稀疏特征系统会使用稀疏存储、特征哈希、在线更新或专门的数据管线；示例没有这些工程优化，也没有真实推荐数据与评测。

## 3. 数学解释

设输入第 `i` 个特征值为 `x_i`，线性权重为 `w_i`，潜向量为 `v_i∈R^k`。二阶 FM 的预测分数是 `ŷ=w₀+Σ_i w_i x_i+Σ_{i<j}〈v_i,v_j〉x_i x_j`。最后一项显式枚举所有特征对会很慢，但可改写为 `1/2 Σ_f[(Σ_i v_{if}x_i)²−Σ_i v_{if}²x_i²]`。源码的 `summed.square() - squared_sum` 正是括号里的量；相减会去掉 `i=j` 的自交互，前面的 `1/2` 去掉 `(i,j)` 与 `(j,i)` 的重复。输出是 logit，点击任务可交给 `BCEWithLogitsLoss`；回归任务也可以改用均方误差，而模型结构不变。
