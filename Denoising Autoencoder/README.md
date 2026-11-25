# DAE：让自编码器从噪声中恢复信息

## 1. 出处、提出者与意义

Pascal Vincent、Hugo Larochelle、Yoshua Bengio 和 Pierre-Antoine Manzagol 在 2008 年 ICML 论文 [《Extracting and Composing Robust Features with Denoising Autoencoders》](https://icml.cc/2008/papers/592.pdf) 中系统提出去噪自编码器。普通自编码器可能学到近似复制输入的捷径；DAE 故意破坏输入，却要求输出恢复原始样本。这样模型必须抓住多个样本共同的结构，而不只是记住某个像素。它是理解自监督表示学习与后来的掩码重建方法的好起点。

## 2. 源码导航与直观解释

打开 [dae.py](dae.py)：`encoder` 把 `[B,16]` 输入压到 `[B,4]`，`decoder` 再还原成 `[B,16]`。`loss` 对干净的 `clean` 现场加高斯噪声，随后比较 `self(corrupted)` 与**原始** `clean`。如果把目标误写为带噪输入，训练就变成复制噪声，失去去噪意义。根目录运行 `python 'Denoising Autoencoder/dae.py'` 会做一次 CPU 前向与反向传播。这里用向量和均方误差讲清原理；论文还研究遮挡等污染方式和深层堆叠。

## 3. 数学解释

令干净样本为 `x`，污染过程为 `q(x̃|x)`，编码器、解码器分别为 `f_θ`、`g_φ`。优化 `E_{x,x̃}‖g_φ(f_θ(x̃))−x‖²`。源码中的 `corrupted = clean + noise_std * randn_like(clean)` 对应从 `q` 抽样；`F.mse_loss(self(corrupted), clean)` 对应右端的重建误差。注意期望同时跨样本与噪声：同一张图每次都可能被破坏得不同，所以学到的特征要对这些扰动稳定。`noise_std` 太小时任务接近普通自编码，太大时原始信息不足、重建也会困难。
