# Latent Diffusion：先压缩图像，再在潜空间扩散

## 1. 出处、提出者与意义

Robin Rombach、Andreas Blattmann、Dominik Lorenz、Patrick Esser、Björn Ommer 在 2021 年提交、2022 年发表于 CVPR 的 [《High-Resolution Image Synthesis with Latent Diffusion Models》](https://arxiv.org/abs/2112.10752) 中提出：直接在每个原始像素上扩散太昂贵，可以先训练自编码器，把图像压到较小的空间特征图，再在该特征图上去噪。论文还用交叉注意力接入文本等条件，让生成方向可控。本目录把“两阶段训练 + 潜空间 DDPM + 文本交叉注意力”完整串起。它是教学版，没有论文的预训练文本编码器、感知/对抗损失、分类器自由引导或大规模权重。

## 2. 源码导航与直观解释

主源码：[latent_diffusion.py](latent_diffusion.py)；本项目同时附带可阅读的本地依赖 [ddpm.py](ddpm.py)，因此无需从其它项目导入模型。`SpatialVAE` 将 `[B,C,H,W]` 压成 `[B,Z,H/4,W/4]`；第一阶段用 `autoencoder_loss` 训练后调用 `freeze_autoencoder()`。第二阶段 `LatentDiffusion.training_loss` 编码图像、在潜图上加噪，并用 `ConditionalLatentUNet` 预测噪声。`TokenConditioner` 把整数文本 token 变成条件向量；`CrossAttention2d` 让每个潜图位置读取相关词语，PAD token 被遮住。`sample` 从潜空间噪声生成，再经 VAE 解码。根目录运行 `python 'Latent Diffusion/latent_diffusion.py'`；示例只演示两阶段的计算，不会生成有语义的图像。

## 3. 数学解释

第一阶段学习 `z=E(x)` 与 `x̂=D(z)`，让重建误差小，同时用 KL 项约束潜变量分布；`SpatialVAE.loss` 实现像素 MSE 加带权 KL。第二阶段不再修改 `E,D`，而对 `z` 用 DDPM 公式 `z_t=√ᾱ_t z₀+√(1−ᾱ_t)ε`，训练 U-Net 预测 `ε`。文本条件的关键式为 `Attention(Q,K,V)=softmax(QKᵀ/√d)V`：`Q` 来自潜图位置，`K,V` 来自文本 token；`CrossAttention2d` 在源码中显式写出点积、掩码和加权求和。采样时倒序去噪得到 `z₀`，然后 `D(z₀)` 还原像素。与像素不同，潜变量通常不限于 `[-1,1]`，所以本地 `DDPM` 设置 `clip_denoised=False`，避免错误裁剪潜值。
