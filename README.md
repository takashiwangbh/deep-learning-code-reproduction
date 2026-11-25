# 深度学习源码复现

这个仓库按论文和模型组织：**一个项目一个文件夹**。每个项目里有可读的源码和三部分中文说明：出处、提出者与意义；源码导航与直观解释；数学解释。建议从多层感知机开始，再按兴趣进入视觉、序列、生成或强化学习项目。

## 项目导航

### 基础

- [Multilayer Perceptron](Multilayer%20Perceptron/README.md)：手写反向传播与完整训练循环。
- [Optimization Algorithms](Optimization%20Algorithms/README.md)：SGD、动量与 Adam。
- [Normalization Layers](Normalization%20Layers/README.md)：BatchNorm 与 LayerNorm。

### 计算机视觉

- [LeNet-5](LeNet-5/README.md)：从卷积和池化开始。
- [AlexNet](AlexNet/README.md)：较深 CNN 与 ReLU。
- [VGG](VGG/README.md)：反复堆叠小卷积核。
- [GoogLeNet](GoogLeNet/README.md)：Inception 多尺度分支。
- [Residual Network](Residual%20Network/README.md)：ResNet-34 的残差连接。
- [U-Net](U-Net/README.md)：编码器与跳跃连接做分割。
- [DenseNet](DenseNet/README.md)：逐层复用前面的特征。
- [MobileNetV2](MobileNetV2/README.md)：轻量化卷积与倒残差。
- [Vision Transformer](Vision%20Transformer/README.md)：把图像块当作 token。
- [YOLOv1](YOLOv1/README.md)：单次前向的目标检测。
- [SimCLR](SimCLR/README.md)：无标签图像的对比学习。

### 序列与语言模型

- [Recurrent Neural Networks](Recurrent%20Neural%20Networks/README.md)：RNN、LSTM 与 GRU 的状态更新。
- [Attention Seq2Seq](Attention%20Seq2Seq/README.md)：翻译中的编码器、解码器与注意力。
- [Transformer](Transformer/README.md)：自注意力与交叉注意力。
- [GPT](GPT/README.md)：因果语言模型与小型训练例子。
- [BERT](BERT/README.md)：双向编码器和掩码预测。
- [Llama-style Language Model](Llama-style%20Language%20Model/README.md)：RoPE、GQA、RMSNorm 与 SwiGLU。
- [LoRA](LoRA/README.md)：低秩参数高效微调。

### 生成、跨模态与图

- [Autoencoder](Autoencoder/README.md)：压缩与重建。
- [Variational Autoencoder](Variational%20Autoencoder/README.md)：潜变量、重参数化与 ELBO。
- [DCGAN](DCGAN/README.md)：生成器与判别器的对抗训练。
- [DDPM](DDPM/README.md)：逐步加噪与去噪。
- [Latent Diffusion](Latent%20Diffusion/README.md)：在潜空间做文本条件扩散。
- [DiT](DiT/README.md)：用 Transformer 预测扩散噪声。
- [Flow Matching](Flow%20Matching/README.md)：学习从噪声到数据的速度场。
- [CLIP](CLIP/README.md)：图文配对的对比学习。
- [GCN](GCN/README.md)：图上的邻居信息聚合。

### 现代模型组件

- [Modern Decoder](Modern%20Decoder/README.md)：把 RoPE、GQA、SwiGLU 与稀疏专家组合起来。
- [Mamba](Mamba/README.md)：选择性状态空间递推。
- [Gated DeltaNet](Gated%20DeltaNet/README.md)：门控遗忘与定向记忆更新。
- [Gated DeltaNet-2](Gated%20DeltaNet-2/README.md)：独立控制记忆的擦除与写入。
- [DPO](DPO/README.md)：用成对偏好优化语言模型。

### 深度强化学习

- [DQN](DQN/README.md)：经验回放与目标网络。
- [PPO](PPO/README.md)：优势估计与裁剪策略更新。

安装依赖见 [requirements.txt](requirements.txt)；在仓库根目录运行 `python check_all.py` 可检查源码。示例使用小张量，没有预训练权重或论文性能指标。
