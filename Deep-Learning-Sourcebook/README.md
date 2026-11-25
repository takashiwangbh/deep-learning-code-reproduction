# 深度学习源码学习路线

这个目录按学习顺序提供**可运行、可读的源码**，从手写反向传播和早期卷积网络，逐步走到 Transformer、扩散模型、现代语言模型组件与强化学习。模型核心结构、注意力、状态递推、损失函数等均在仓库里实现；`import torch` 是使用张量和自动微分，不是导入现成模型。

这些文件是论文机制的教学实现。随机输入的运行检查证明形状、梯度与部分关键性质正确，不等于复现论文的训练数据、预训练权重或榜单成绩。每个主题 README 都列出论文来源和可见的简化。

## 从哪里开始

| 阶段 | 阅读顺序 | 要掌握的机制 |
| --- | --- | --- |
| 1 基础 | [foundations](foundations/README.md) | 手写反向传播、SGD/Adam、归一化、训练循环 |
| 2 视觉 | [vision](vision/README.md) | LeNet → AlexNet → VGG → GoogLeNet → [已有 ResNet-34](../Residual%20Network/ResNet-34%20.py) → U-Net → DenseNet → MobileNetV2 → ViT；YOLOv1 检测与 SimCLR 自监督 |
| 3 序列 | [sequence](sequence/README.md) | RNN/LSTM/GRU → Seq2Seq 注意力 → Transformer → GPT/BERT → LoRA 与现代解码器 |
| 4 生成与跨模态 | [generative](generative/README.md) | AE/VAE → DCGAN → DDPM → 潜空间扩散 → DiT → Flow Matching；另有 CLIP 和 GCN |
| 5 现代组件 | [modern](modern/README.md) | RoPE/GQA/MoE、Mamba、2025/2026 Gated DeltaNet、DPO |
| 6 强化学习 | [reinforcement](reinforcement/README.md) | DQN 经验回放/目标网络、PPO GAE/裁剪目标 |

先完成 `foundations/mlp_numpy.py`，再选一个主题按目录 README 顺序阅读。每读完一个模型，先运行文件末尾的例子，再修改一个超参数、检查输出形状与梯度。`sequence/train_toy_gpt.py` 是带有小型内置语料的完整训练示例。

## 安装与运行

建议使用 Python 3.10+ 的独立环境。PyTorch 的安装命令可能因操作系统和 GPU 而异，可先看 [官方安装页](https://pytorch.org/get-started/locally/)；然后安装本目录依赖：

```bash
python -m pip install -r Deep-Learning-Sourcebook/requirements.txt
python Deep-Learning-Sourcebook/foundations/mlp_numpy.py
python Deep-Learning-Sourcebook/foundations/train_mlp.py
python Deep-Learning-Sourcebook/check_all.py
```

`check_all.py` 逐目录运行小张量检查；全部检查都不需要下载数据或权重。建议至少有数百 MB 可用内存。每个主题目录还提供单文件执行命令。

## 源码与来源

这里的实现根据原论文公式与架构重新编写，没有直接复制第三方 GitHub 代码。每个主题 README 包含原论文链接；原仓库中的 [ResNet-34](../Residual%20Network/README.md) 保持独立。若以后引入第三方代码，请按 [贡献说明](CONTRIBUTING.md) 保留原作者与许可证。整个仓库的许可证见 [LICENSE.md](../LICENSE.md)。

## 范围

这是持续扩展的学习库，而不是对每年所有论文的穷尽收录。当前覆盖深度学习史上具有代表性的网络家族，以及较新的稀疏专家、状态空间、流匹配与偏好优化机制。新增论文时，请保持“完整源码、数学对应、可运行验证和来源说明”四项标准。
