# 循环神经网络：RNN、LSTM、GRU

## 1. 出处、提出者与意义

这个文件夹放在一起的是三个紧密相连的循环网络，而不是三份互不相关的模型。最基础的简单 RNN 可从 Jeffrey L. Elman 1990 年的 [《Finding Structure in Time》](https://doi.org/10.1207/s15516709cog1402_1) 了解：它把上一时刻的隐藏状态带到下一时刻，让网络能够处理顺序。Sepp Hochreiter 与 Jürgen Schmidhuber 1997 年提出 [LSTM](https://doi.org/10.1162/neco.1997.9.8.1735)，用记忆单元和门控制信息保留，缓解普通 RNN 很难学到长距离依赖的问题。Kyunghyun Cho、Bart van Merrienboer、Caglar Gulcehre、Dzmitry Bahdanau、Fethi Bougares、Holger Schwenk 与 Yoshua Bengio 2014 年的 [RNN Encoder–Decoder](https://arxiv.org/abs/1406.1078) 引入了常用的 GRU 门控结构，用更少的门完成类似的信息筛选。

读完这三个模型，就能理解序列模型最重要的起点：**同一组参数反复处理每个时间步，状态承担“记住此前内容”的任务**。后面的注意力和 Transformer 可以视为对“如何读取过去信息”的不同回答。本项目只演示结构与计算，不附带论文训练数据或预训练权重。

## 2. 源码导航与直观解释

打开 [recurrent.py](recurrent.py)，按 `RNNCell → LSTMCell → GRUCell → RecurrentLanguageModel` 阅读。前三个类各实现一次时间步的状态更新；最后一个类把词 ID 映射成向量，逐词调用所选单元，再把每一步状态映射成词表分数。输入 `tokens` 是 `[批量 B, 长度 T]` 的整数，输出 `logits` 是 `[B,T,词表 V]`；LSTM 还返回 `(hidden, cell)` 两份状态，其余模型只返回 hidden。

可以把隐藏状态想成一张不断改写的便签。普通 RNN 每步几乎重写整张便签；LSTM 分别决定“丢掉多少旧内容、写入多少新内容、给外界看多少”；GRU 用较少的门把旧状态与新候选混合。代码没有调用 `nn.LSTM` 或 `nn.GRU`，门与循环都在文件中。仓库根目录运行 `python 'Recurrent Neural Networks/recurrent.py'`，会打印三种网络的输出形状；该随机输入只检查计算是否可运行。

## 3. 数学解释

设第 `t` 步输入词向量为 `xₜ`，上一时刻状态为 `hₜ₋₁`。**简单 RNN** 在 `RNNCell.forward` 计算 `hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁ + b)`。`Wₓ` 处理新输入，`Wₕ` 处理旧状态；两个线性层的结果相加后经过 `tanh`。训练时梯度要沿许多时间步反传，反复乘以导数可能使较早的信息难以保留。

**LSTM** 在 `LSTMCell.forward` 一次算出四组值：输入门 `iₜ`、遗忘门 `fₜ`、候选内容 `gₜ`、输出门 `oₜ`。门的 `sigmoid` 值在 0 到 1 之间，可以理解为“放行比例”。核心更新是 `cₜ = fₜ⊙cₜ₋₁ + iₜ⊙gₜ`，`hₜ = oₜ⊙tanh(cₜ)`；`⊙` 表示逐元素相乘。`c` 这条加法路径就是较长记忆的通道：若 `f` 接近 1，旧记忆便能继续流动。

**GRU** 在 `GRUCell.forward` 计算重置门 `rₜ`、更新门 `zₜ` 和候选状态 `nₜ`，再算 `hₜ = zₜ⊙hₜ₋₁ + (1-zₜ)⊙nₜ`。这里 `z` 接近 1 表示多保留旧状态，接近 0 表示多采用新内容。最后在 `RecurrentLanguageModel.forward` 的 Python 循环里，前一步返回的状态会作为下一步输入，这就是上面所有带下标 `t-1` 的公式在源码中的位置。
