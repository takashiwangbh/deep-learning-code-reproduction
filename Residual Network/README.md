# ResNet-34 残差网络

论文：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)（2015）。

当前目录提供带中文注释的 [`ResNet-34 .py`](ResNet-34%20.py)。核心是 `BasicBlock` 中的 `F(x)+x` 跳跃连接；当空间尺寸或通道数改变时，代码使用 1×1 卷积投影 `identity`，使两条路径形状一致。四个 stage 的残差块数量为 `[3,4,6,3]`。

从仓库根目录运行一个随机输入的形状和反向传播检查：

```bash
python 'Residual Network/ResNet-34 .py'
```

这里只提供 ResNet-34 源码；目录中暂未包含 ResNet-101 和论文笔记 PDF。其它视觉模型的学习顺序见 [源码学习路线](../Deep-Learning-Sourcebook/vision/README.md)。随机张量检查不代表论文性能复现。

