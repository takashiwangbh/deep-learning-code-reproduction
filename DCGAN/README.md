# DCGAN：用两个卷积网络对抗生成图像

## 1. 出处、提出者与意义

Alec Radford、Luke Metz 与 Soumith Chintala 于 2015 年发表 [《Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks》](https://arxiv.org/abs/1511.06434)。GAN 的“生成器对判别器”思想已有先例，DCGAN 的贡献是给图像任务提出一套有效的深层卷积结构约束，让无标签图像生成与特征学习更稳定、更容易复用。它说明生成器不必记住训练图片，而可以从随机向量逐级构造新的空间结构。此目录实现了生成器、判别器与两者各自的损失，不包含真实数据集和训练好的权重。

## 2. 源码导航与直观解释

源码：[dcgan.py](dcgan.py)。`Generator` 从随机向量 `z:[B,Z]` 出发，连续用转置卷积把 `1×1` 特征图放大为 `64×64` RGB 图像；最后 `tanh` 将像素压到 `[-1,1]`。`Discriminator` 走相反方向，把图像缩小并输出真假 **logit** `[B]`。`discriminator_loss` 让真实图像得高分、生成图像得低分；`generator_loss` 让生成图像骗过判别器。训练判别器时，`fake.detach()` 很关键：这一轮只改判别器，不误改生成器。从仓库根目录运行 `python DCGAN/dcgan.py`，会完成一次小模型的双向反向传播。

## 3. 数学解释

记 `G(z)` 为生成器、`D(x)` 为判别器给“真实”的概率。经典对抗目标为 `min_G max_D E_{x∼data}[logD(x)]+E_{z∼prior}[log(1−D(G(z)))]`。两个网络方向相反：`D` 想把真假分开，`G` 想让假图像更像真图像。源码中的判别器损失等价于对真实样本标签 1、生成样本标签 0 分别做二元交叉熵；生成器使用非饱和目标 `−log D(G(z))`，在早期较容易获得有效梯度。`binary_cross_entropy_with_logits` 接收未经过 sigmoid 的 logit，内部稳定地完成概率计算。真实图像须归一化到 `[-1,1]`，与 `G` 的输出范围一致；否则判别器可能只靠数值范围区分真假。
