# 生成模型、图文对齐与图网络源码

本目录的模型均直接用 PyTorch 张量、自动微分和基础网络层实现；不会调用 `torchvision.models`、`transformers`、`diffusers` 等现成模型。源码是为了读懂论文机制的**小尺寸教学实现**，随机输入可以跑通前向和反向传播；没有训练权重，也不承诺复现论文指标。每个 `.py` 文件都可作为脚本运行；后面的模型会复用本目录前面模型的源码。

## 学习路线

| 顺序 | 文件 | 论文与核心思想 | 先看什么 |
| --- | --- | --- | --- |
| 1 | [`autoencoder.py`](autoencoder.py) | [Hinton & Salakhutdinov, 2006](https://doi.org/10.1126/science.1127647)：编码器 → 瓶颈 → 解码器 | `forward` 与 MSE 重建损失 |
| 2 | [`vae.py`](vae.py) | [Auto-Encoding Variational Bayes, 2013](https://arxiv.org/abs/1312.6114)：随机潜变量与 ELBO | `reparameterize`、`elbo_loss` |
| 3 | [`dcgan.py`](dcgan.py) | [DCGAN, 2015](https://arxiv.org/abs/1511.06434)：转置卷积生成图像、卷积判别真假 | 两个独立损失和 `fake.detach()` |
| 4 | [`ddpm.py`](ddpm.py) | [Denoising Diffusion Probabilistic Models, 2020](https://arxiv.org/abs/2006.11239)：加噪与反向去噪 | `q_sample`、`training_loss`、`p_sample` |
| 5 | [`latent_diffusion.py`](latent_diffusion.py) | [High-Resolution Image Synthesis with Latent Diffusion Models, 2021/2022](https://arxiv.org/abs/2112.10752)：压缩图像后在潜空间做文本条件扩散 | `SpatialVAE`、`CrossAttention2d`、`LatentDiffusion` |
| 6 | [`dit.py`](dit.py) | [Scalable Diffusion Models with Transformers, 2022](https://arxiv.org/abs/2212.09748)：patch Transformer 预测扩散噪声 | `DiTBlock` 的 adaLN 与门控 |
| 7 | [`flow_matching.py`](flow_matching.py) | [Flow Matching for Generative Modeling, 2022](https://arxiv.org/abs/2210.02747)：学习连续速度场 | `interpolation`、`training_loss`、`sample` |
| 8 | [`clip.py`](clip.py) | [CLIP, 2021](https://arxiv.org/abs/2103.00020)：图文双编码器与批内对比学习 | `CLIP.forward`、`contrastive_loss` |
| 9 | [`gcn.py`](gcn.py) | [GCN, 2016](https://arxiv.org/abs/1609.02907)：邻接矩阵归一化、邻居特征聚合 | `normalize_adjacency`、`GraphConv` |

## 数学与代码对照

- **AE**：`z = encoder(x)`、`x_hat = decoder(z)`，目标为 `MSE(x_hat, x)`。瓶颈维度小于输入维度。
- **VAE**：`reparameterize` 写出 `z = μ + exp(½ log σ²) · ε`，`ε ~ N(0,I)`；`elbo_loss` 计算 `BCEWithLogits(x_hat,x) + KL(q(z|x) || N(0,I))`。重建项与 KL 都按 batch 平均。
- **DCGAN**：生成器从 `[B, Z, 1, 1]` 逐级上采样到 `[B, 3, 64, 64]`；判别器反向压缩到真假 logit。训练判别器时对 `G(z)` 使用 `.detach()`；训练生成器时不分离梯度。真实图像应归一化到 `[-1,1]`。
- **DDPM**：`q_sample` 用 `x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε` 一次得到任意时间步；网络训练时预测 `ε`；`p_sample` 用预测的 `x₀` 计算高斯后验均值并采样。`t=0` 不再加随机噪声。`TinyUNet` 是讲解用的两级 U-Net。
- **潜空间扩散**：`SpatialVAE` 将 `[B,C,H,W]` 图像编码为 `[B,Z,H/4,W/4]` 空间潜变量；先独立训练它，再冻结。`LatentDiffusion.training_loss` 在编码后的 `z` 上调用 DDPM 的加噪与噪声预测；`CrossAttention2d` 显式计算 `softmax(Q_image K_textᵀ / √d) V_text`，文本 PAD 位置被掩盖。`sample` 在潜空间逐步去噪，再经 VAE 解码。由于潜变量没有 `[-1,1]` 边界，调用 `DDPM(..., clip_denoised=False)`。
- **DiT**：卷积把输入切为 patch，时间嵌入产生条件向量；每个 Transformer 块的 `shift/scale/gate` 对注意力和 MLP 做 adaLN 调制。最终恢复为输入的图像形状。它只实现噪声预测骨干；可作为 DDPM 的 `denoiser`，输入尺寸需匹配。
- **Flow Matching**：从高斯噪声 `x₀` 到真实样本 `x₁` 的直线路径 `x_t=(1-t)x₀+t x₁`，目标速度 `x₁-x₀`。`training_loss` 回归该速度；`sample` 从噪声出发用 Euler 步进积分。这里复用 `ddpm.py` 中的时间条件 U-Net，但输出含义从“噪声”变成“速度”。
- **CLIP**：图像和文本各自编码并做 L2 归一化，计算所有图文配对的余弦相似度矩阵；用两个方向的交叉熵平均训练。一个 batch 中相同下标的图和文必须配对。`lengths` 指向句末有效 token。
- **GCN**：`normalize_adjacency` 构造 `D̃⁻¹ᐟ²(A+I)D̃⁻¹ᐟ²`，`GraphConv` 对特征做线性变换后聚合邻居；训练时只对有标签节点计算交叉熵。

## 运行

需 Python 3.10+ 和 PyTorch。安装方式见 [PyTorch 官方页面](https://pytorch.org/get-started/locally/)。在本目录执行：

```bash
python autoencoder.py
python vae.py
python dcgan.py
python ddpm.py
python latent_diffusion.py
python dit.py
python flow_matching.py
python clip.py
python gcn.py
python test_models.py
```

这些命令只使用随机张量，CPU 可运行，不下载数据。若要做真正训练，需为目标数据准备数据加载器、优化器、完整训练循环、检查点与评估；这里保留模型和损失函数的完整源码作为起点。

### 潜空间扩散的两阶段用法

`latent_diffusion.py` 的主函数只演示一次前向、反向与采样；随机初始化的自编码器不能产生有意义的图像。真实训练应先用图像（归一化到 `[-1,1]`）充分优化 `model.autoencoder_loss(images)`，优化器只接收 `model.autoencoder.parameters()`。完成后调用 `model.freeze_autoencoder()`；第二阶段优化 `model.conditioner.parameters()` 与 `model.diffusion.denoiser.parameters()`，损失为 `model.training_loss(images, token_ids)`。采样时传入同一套词表产生的 token IDs：`model.sample(token_ids)`。

本教学版与原论文的**具体差异**：空间 VAE 只有两级卷积压缩，第一阶段使用像素 MSE 与 KL，不包含原论文的感知损失/对抗训练；文本编码器是可训练的 token 与位置嵌入，而不是预训练文本模型，也不附带 tokenizer；U-Net 只有一个瓶颈交叉注意力层；潜变量缩放固定为构造参数 `latent_scale`（默认 1.0），没有从训练集统计估计；不含 classifier-free guidance、论文规模的网络、长时间训练或权重。它保留了“先训练压缩模型 → 在空间潜变量上扩散 → 通过交叉注意力读取条件 → 解码”的关键计算路径。

## 来源与许可

以上文件依据表中论文的公式与架构独立编写，没有复制外部 GitHub 仓库代码；本仓库的许可证见根目录 `LICENSE.md`。论文链接是理论来源，不代表论文作者对本实现提供保证。
