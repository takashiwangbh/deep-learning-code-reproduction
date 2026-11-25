# CVAE：按条件生成多种合理结果

## 1. 出处、提出者与意义

Kihyuk Sohn、Xinchen Yan 与 Honglak Lee 在 NIPS 2015 论文 [《Learning Structured Output Representation using Deep Conditional Generative Models》](https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html) 中研究条件潜变量生成。普通 VAE 学 `p(x)`；CVAE 学 `p(x|y)`：给定类别或其它条件 `y`，仍可通过不同潜变量 `z` 产生多种合理样本。原论文面向结构化预测；这里用类别标签和短向量说明条件先验、条件后验与重参数化的核心。

## 2. 源码导航与直观解释

在 [cvae.py](cvae.py) 先看 `embedding` 如何把整数标签 `[B]` 变成向量；`posterior` 接受 `[x,y]` 形成 `q(z|x,y)`，`prior` 只接受 `y` 形成 `p(z|y)`，`decoder` 从 `[z,y]` 预测输出 logits。训练时已有真实 `x`，所以可以从后验采样；生成时没有 `x`，`sample` 必须改从条件先验采样。输入 `x` 为 `[B,16]` 且数值在 `[0,1]`，输出也是 `[B,16]`。运行 `python 'Conditional Variational Autoencoder/cvae.py'`。这里是 Bernoulli 输出的小向量模型，未复现论文的大图结构预测网络。

## 3. 数学解释

目标是最大化条件 ELBO：`E_{q(z|x,y)}[log p(x|z,y)] − KL(q(z|x,y) || p(z|y))`。`loss` 返回其相反数：第一项用 `binary_cross_entropy_with_logits`，第二项用两个对角高斯的解析 KL。若 `q=N(μq,σq²)`、`p=N(μp,σp²)`，每一维 KL 为 `0.5[log(σp²/σq²)+(σq²+(μq−μp)²)/σp²−1]`；源码中的 `plv-qlv` 等项逐项对应。采样 `z=μq+σq ε`、`ε∼N(0,I)` 把随机性移到不依赖参数的噪声上，因此梯度能穿过 `μq` 和 `σq`。预测时则把式中的后验参数换成先验参数。
