# 深度学习论文源码复现

面向初学者，按论文与核心机制阅读深度学习源码。模型结构、注意力、损失与训练更新都写在仓库中；使用 PyTorch 基础运算，但不以导入现成模型代替实现。

## 从基础读到现代模型

新建的 [Deep-Learning-Sourcebook](Deep-Learning-Sourcebook/README.md) 包含基础网络、经典视觉模型与 YOLOv1/SimCLR、RNN/Transformer/GPT/BERT、生成与潜空间扩散模型、CLIP/GCN、现代语言模型组件、DQN/PPO。每个主题都有中文学习顺序、论文链接、公式与代码的对应、运行说明。完整的单目录运行检查：

```bash
python -m pip install -r Deep-Learning-Sourcebook/requirements.txt
python Deep-Learning-Sourcebook/check_all.py
```

已有的 [Residual Network](Residual%20Network/README.md) 保留原来的 ResNet-34 注释源码。建议先读 `Deep-Learning-Sourcebook/foundations`，再读 LeNet、AlexNet、VGG、ResNet 和后续模型。

这些是供学习的源码实现，并未附带大规模数据集、预训练权重或论文性能复现实验。许可证见 [LICENSE.md](LICENSE.md)。
