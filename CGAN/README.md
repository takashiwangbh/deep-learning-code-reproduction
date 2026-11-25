# CGAN：给生成过程一个条件

## 1. 出处、提出者与意义

Mehdi Mirza 与 Simon Osindero 在 2014 年 [《Conditional Generative Adversarial Nets》](https://arxiv.org/abs/1411.1784) 中提出把额外信息 `y` 同时交给生成器和判别器。普通 GAN 只需像总体数据分布；CGAN 还要求“在给定标签时像对应类别”。论文用类别控制手写数字生成，也展示了多模态条件建模的可能性。之后许多图像翻译和文本引导生成方法都沿用这个条件思想。

## 2. 源码导航与直观解释

[cgan.py](cgan.py) 中 `ConditionalGenerator.forward` 把随机噪声 `[B,8]` 与标签的 one-hot 向量拼接，输出 `[B,16]`；`ConditionalDiscriminator.forward` 则把同一个标签与真假样本拼接后判断。`losses` 让 D 比较“真实数据 + 正确标签”和“生成数据 + 同标签”，G 试图在该标签下骗过 D。根目录运行 `python CGAN/cgan.py`。标签在本例是 3 类整数；实际使用应有该类别的真实样本，且分别更新两个网络。

## 3. 数学解释

无条件 GAN 的 `G(z)` 和 `D(x)` 改为 `G(z,y)` 与 `D(x,y)`。条件目标为 `min_G max_D E_{(x,y)∼p_data} log D(x,y)+E_{z,y} log(1−D(G(z,y),y))`。代码用 logits 的 `softplus` 写交叉熵，并对 G 使用非饱和目标 `−log D(G(z,y),y)`。`F.one_hot` 只负责表示条件，不提供生成能力；能力来自模型学习标签如何影响输出，且 D 也据此判断样本是否符合条件。若训练时把真假样本的 `y` 搞乱，模型就学不到正确的条件分布。
