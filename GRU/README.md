# GRU：更紧凑的门控循环单元

## 1. 出处、提出者与意义

Kyunghyun Cho 等人在 2014 年 [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) 中引入后来被称为 GRU 的门控循环单元。它以更新门和重置门控制记忆，不单独维护 LSTM 的细胞状态，因此结构更紧凑，也常用于编码器、解码器与时序预测。

## 2. 源码导航与直观解释

[gru.py](gru.py) 里 `GRUCell.forward` 先求 `reset` 和 `update`，再求候选隐藏状态，最后在候选与旧状态之间逐维插值。`GRU.forward` 把这套计算应用于每一个时间步。输入 `[B,T,D]`，输出 `[B,T,H]` 与末状态 `[B,H]`。运行 `python GRU/gru.py`。这里用基础线性层写出门控，而不是调用 `nn.GRU`。候选项采用“先对隐藏状态做线性变换，再乘重置门”的常见变体；原论文的乘门位置略有不同。

## 3. 数学解释

重置门 `rₜ=σ(Wᵣxₜ+Uᵣhₜ₋₁)`，更新门 `zₜ=σ(W_zxₜ+U_zhₜ₋₁)`；候选 `nₜ=tanh(Wₙxₜ+rₜ⊙Uₙhₜ₋₁)`，最终 `hₜ=(1−zₜ)⊙nₜ+zₜ⊙hₜ₋₁`。`r` 小时，候选较少使用旧记忆；`z` 大时，最终状态主要保留旧记忆。源码的 `input_new` 与 `hidden_new` 分别实现候选中的两项。

若把所有权重清零，更新门等于 0.5，候选等于 0，输入的旧隐藏状态会被保留一半；测试脚本正用这个边界情形验证插值式。GRU 只有一个持久状态 h，没有 LSTM 中独立的 c。对时间很长的序列，门能帮助保留信息，却不保证任何任务上都优于 LSTM。
