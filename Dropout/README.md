# Dropout：训练时随机失活，测试时使用完整网络

## 1. 出处、提出者与意义

Nitish Srivastava、Geoffrey Hinton、Alex Krizhevsky、Ilya Sutskever、Ruslan Salakhutdinov 在 2014 年的 [《Dropout: A Simple Way to Prevent Neural Networks from Overfitting》](https://jmlr.org/papers/v15/srivastava14a.html) 系统研究了这一正则化方法。训练时随机隐藏一些神经元及其连接，网络就难以只依赖某几个特征之间的固定配合；测试时保留全部神经元。Dropout 因简单、可插入多种架构，成为深度学习早期最有影响力的通用训练机制之一。

## 2. 源码导航与直观解释

[dropout.py](dropout.py) 的 `inverted_dropout` 为每个输入元素采样一个 Bernoulli 保留掩码；训练时被丢弃的位置为零，被保留的位置乘 `1/(1-p)`。`DropoutMLP` 把它接在 ReLU 隐藏层之后，`model.train()` 会随机丢弃，`model.eval()` 直接返回原隐藏向量。输入 `[B,F]`，输出 `[B,C]` 类别 logits。根目录运行 `python Dropout/dropout.py` 观察前向和梯度。示例按元素随机失活，不包含卷积网络常见的通道 Dropout 或随机深度。

## 3. 数学解释

令 `m_i∼Bernoulli(1-p)`，倒置式 Dropout 的输出 `y_i=x_i m_i/(1-p)`。其期望 `E[y_i]=x_i(1-p)/(1-p)=x_i`，因此测试阶段直接用 `y_i=x_i`，不再另乘保留率。`inverted_dropout` 中的 `rand_like >= p` 对应保留掩码，除以 `1-p` 保持训练与推理激活量级相近。随机掩码只用于训练，每个 batch 可不同；它改变的是训练目标的随机扰动，不是一个可学习的权重层。`p` 必须小于 1，否则缩放分母为零。
