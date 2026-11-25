# AlexNet：把深卷积网络带入大规模图像分类

## 1. 出处、提出者与意义

Alex Krizhevsky、Ilya Sutskever 和 Geoffrey E. Hinton 在 2012 年发表 [《ImageNet Classification with Deep Convolutional Neural Networks》](https://proceedings.neurips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)。他们在大规模 ImageNet 分类任务上训练较深的卷积网络，展示了数据、GPU 计算和合适的激活与正则化可以让深网络明显提升图像识别。AlexNet 的重要性不仅在于层数，还在于把 ReLU、局部响应归一化、最大池化和 dropout 组织成可训练的整体，推动后续视觉模型研究。

## 2. 源码导航与直观解释

打开 [alexnet.py](alexnet.py) 的 `AlexNet`：`features` 是五个卷积层，从图像提取边缘、纹理到物体部件；`avgpool` 把特征图统一到指定大小；`classifier` 是三层全连接分类头。输入示例 `[2,3,96,96]`，输出 `[2,10]` logits。先逐层读 `features` 中卷积、ReLU、LRN 和池化的顺序，再看 `Dropout(0.5)` 如何在训练时随机屏蔽一部分分类头神经元。`groups=2` 表示分组连接，教学版用单机运算表达原论文两块 GPU 上的部分分组结构。

从仓库根目录运行 `python "AlexNet/alexnet.py"`。默认网络很大；文件末尾把通道宽度、全连接层和池化尺寸缩小，仅供 CPU 演示。随机 logits 不是训练结果。

## 3. 数学解释

卷积的一般尺寸公式是 `H_out=⌊(H+2p−k)/s⌋+1`，其中 `H` 为输入高、`k` 为卷积核、`p` 为填充、`s` 为步长。首层 `11×11`、步长 4，因此很快缩小图像；后续卷积在更小的特征图上加工。ReLU 为 `max(0,x)`，正数保持不变、负数变零，避免深层中大量单元落在饱和区。最大池化在每个局部窗口取最大值，保留最强响应。dropout 可写为 `h'=m⊙h/(1-p)`，其中 `m` 是随机 0/1 掩码、`p` 是丢弃概率；源码由 `nn.Dropout` 完成。它帮助分类头减少对某几个固定特征的依赖。
