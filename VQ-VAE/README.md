# VQ-VAE：把连续特征变成离散代码

## 1. 出处、提出者与意义

Aaron van den Oord、Oriol Vinyals 和 Koray Kavukcuoglu 在 2017 年 [《Neural Discrete Representation Learning》](https://arxiv.org/abs/1711.00937) 中提出 VQ-VAE。VAE 通常用连续高斯潜变量；VQ-VAE 让每个空间位置从有限个代码向量中选一个，得到可作为离散 token 使用的表示。离散码本使压缩、生成先验建模与跨模态表示更容易衔接，也缓解强解码器忽略潜变量的问题。

## 2. 源码导航与直观解释

打开 [vq_vae.py](vq_vae.py)：`VQVAE.encoder` 把 `[B,1,16,16]` 压到 `[B,8,4,4]`；`VectorQuantizer` 为每个位置比较 16 个码本向量的平方距离，得到 `[B,4,4]` 离散索引；`decoder` 再还原图像。根目录执行 `python VQ-VAE/vq_vae.py`。重点阅读 `z_st` 一行：前向喂给解码器的是最近的码本向量，反向却让梯度直接流回编码器。代码展示离散瓶颈与重建训练；论文中的自回归代码先验没有放进这个小例子。

## 3. 数学解释

编码器给出 `z_e(x)`，码本为 `{e_k}`，每个位置选择 `k*=argmin_k ‖z_e−e_k‖²`，量化结果 `z_q=e_{k*}`。最近邻选择不可导，所以用 straight-through 变量 `z_st=z_e+stopgrad(z_q−z_e)`：前向数值等于 `z_q`，对 `z_e` 的反向导数却是 1。总损失是 `L_recon + ‖stopgrad(z_e)−z_q‖² + β‖z_e−stopgrad(z_q)‖²`。第二项只拉动码本，第三项只让编码器输出靠近被选中的码本；源码分别叫 `codebook_loss` 和 `commitment_loss`，默认 `β=0.25`。这三个目标缺一不可：仅有重建误差时码本更新方式与编码器稳定性都可能出问题。
