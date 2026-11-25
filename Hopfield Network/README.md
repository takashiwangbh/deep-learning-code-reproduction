# Hopfield Network：可按线索找回记忆的循环网络

## 1. 出处、提出者与意义

John J. Hopfield 在 1982 年的 [《Neural networks and physical systems with emergent collective computational abilities》](https://pmc.ncbi.nlm.nih.gov/articles/PMC346238/) 中把对称连接的神经元系统解释为一种内容寻址记忆：给出一幅有噪声的模式，网络可沿能量下降的方向回到某个已存模式。与普通分类器“输入后一次给答案”不同，它反复更新自身状态，展示了神经网络如何通过动力系统的稳定点存储信息。这里实现论文的**经典二值离散 Hopfield 网络**；它与后来用注意力实现的现代 Hopfield 网络不是同一个架构。

## 2. 源码导航与直观解释

[hopfield.py](hopfield.py) 的 `HopfieldMemory.__init__` 将若干 `[P,N]` 的 `-1/+1` 模式用 Hebb 规则写入 `N×N` 权重矩阵，并把对角线设为零；`energy` 计算一批状态的能量；`recall` 按固定顺序一次只更新一个神经元，做若干轮。可以把记忆想成地形里的低谷，输入被翻错几位后，反复更新让状态向附近低谷移动。根目录运行 `python 'Hopfield Network/hopfield.py'`，会打印受损模式恢复前后的能量与最终状态。示例只存一幅小模式；容量、串扰、随机异步顺序等真实问题需用更多模式研究。

## 3. 数学解释

对存储模式 `ξ^μ`，权重 `W_ij=(1/N)Σ_μ ξ_i^μ ξ_j^μ`（`i≠j`），`W_ii=0`。状态 `s∈{-1,+1}^N` 的能量 `E(s)=-1/2 Σ_{i,j} W_ij s_i s_j`；源码的矩阵乘法与此式相同。更新第 `i` 位时，先算局部场 `h_i=Σ_j W_ij s_j`，再令 `s_i=sign(h_i)`；当 `h_i=0` 时保留原值。因为 `W` 对称且无自连接，单独更新一位不会增加能量；这解释了为何网络能收敛到稳定模式，也说明同步更新所有位并不具有同样的单步保证。
