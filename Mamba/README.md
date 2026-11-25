# Mamba（选择性状态空间模型）

## 1. 出处、提出者与意义

Albert Gu 与 Tri Dao 在 [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)（2023）中提出 Mamba。标准 Transformer 注意力要比较序列中的 token 对，序列变长时计算与存储开销增长明显。状态空间模型改为维护一份随时间更新的状态；Mamba 的关键是让更新参数由当前输入决定，从而选择保留或忘记哪些信息。

本项目适合在读完 RNN 和 Transformer 后学习：它像 RNN 一样按时间更新状态，却使用专门设计的参数化处理长序列。源码强调算法式本身，不包含论文为 GPU 设计的融合算子与并行扫描，因此运行速度不能拿来比较真实 Mamba 系统。

## 2. 源码导航与直观解释

[mamba_selective_ssm.py](mamba_selective_ssm.py) 中的 `MambaBlock` 是核心。先看 `in_proj` 把输入分成数据支路与门控支路；`conv` 只结合当前位置和过去的局部信息；`x_proj` 产生随输入变化的 `Δ、B、C`；然后 `for t in range(length)` 明确写出每一步状态如何更新。`MambaLanguageModel` 将多个块叠起来，把 token ID `[B,T]` 变成词表分数 `[B,T,V]`。

从仓库根目录运行 `python 'Mamba/mamba_selective_ssm.py'`。脚本使用很小的随机 token 和网络宽度，执行一次下一个 token 预测的反向传播。读者可以把第 4 个之后的输入改掉，验证前 4 个位置的输出不变：因果卷积和状态递推只能读取过去。

## 3. 数学解释

把第 `t` 个时间步的输入记为 `uₜ`、隐藏状态记为 `hₜ`。连续状态矩阵 `A` 在源码中设为负值；输入产生正的步长 `Δₜ`，所以离散衰减 `exp(ΔₜA)` 在 0 与 1 之间。核心更新是 `hₜ = exp(ΔₜA) ⊙ hₜ₋₁ + ΔₜBₜ ⊙ uₜ`，输出是 `yₜ = Cₜhₜ + Duₜ`。`Bₜ` 决定写入什么，`Cₜ` 决定读出什么，`Δₜ` 控制旧状态衰减与新信息写入的尺度；下标 `t` 表明它们会随当前输入变化。

源码里的 `state` 形状是 `[批量, 内部通道, 状态维度]`；`delta_a`、`delta_b_u` 分别对应上式两项。每一步只更新固定大小的状态，因此递推计算随序列长度 `T` 线性增长。输出再乘 `SiLU(gate)` 并经过 `out_proj` 返回原特征维度。Python 循环展示的是同一递推数学关系，并非论文中的高吞吐并行实现。
