# Performer：用正随机特征改写注意力

## 1. 出处、提出者与意义

Krzysztof Choromanski、Valerii Likhosherstov、David Dohan 等人在 2020 年提出 [《Rethinking Attention with Performers》](https://arxiv.org/abs/2009.14794)。普通 softmax 注意力先形成长度 `T` 的两两相似度矩阵。Performer 用 FAVOR+ 的正交随机特征近似指数核，再利用乘法结合律绕开 `T×T` 矩阵，使长序列注意力可以按长度线性扩展。本项目实现核心的**正随机特征 + 因果递推**；随机矩阵使用普通高斯采样而非论文的正交构造，故是教学近似。

## 2. 源码导航与直观解释

[performer.py](performer.py) 的 `FavorAttention._feature` 把每个 Q/K 向量投到多个随机方向，经指数变成非负特征 `φ(q),φ(k)`。`forward` 逐词更新两个状态：`kv_state` 累积 `φ(k)⊗v`，`k_state` 累积 `φ(k)`；当前 query 与这两个状态分别相乘，得到分子和归一化分母。`Performer` 包上嵌入与输出层，输入 `[B,T]` token，返回 `[B,T,V]` logits。根目录运行 `python Performer/performer.py`。

这里显式逐 token 扫描，便于确认未来词不进入过去状态；大型训练常用并行扫描与优化内核。固定随机投影、有限特征数带来近似误差，没有预训练权重或论文指标。

## 3. 数学解释

softmax 分子中的 `exp(q·k/√d)` 可由非负映射近似为 `φ(q)ᵀφ(k)`。于是第 `t` 个因果输出可写成 `o_t=[φ(q_t)ᵀΣ_{i≤t}φ(k_i)v_iᵀ]/[φ(q_t)ᵀΣ_{i≤t}φ(k_i)]`。源码的 `kv_state` 是分子内的矩阵和，`k_state` 是分母内的向量和；每来一个 token 只更新一次，既不存 `T×T` 分数，也不读取未来。随机特征数 `m` 越大通常越接近精确 softmax，但状态和乘法成本也随 `m` 增长。为避免指数溢出，代码对 query 单独平移、对 key 只用**当前及过去**的最大值平移；最大值更新时同步缩放旧状态。这个细节保证未来 key 不会通过数值归一化影响过去的输出。
