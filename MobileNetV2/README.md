# MobileNetV2：在手机等设备上节省卷积计算

## 1. 出处、提出者与意义

Mark Sandler、Andrew Howard、Menglong Zhu、Andrey Zhmoginov 和 Liang-Chieh Chen 在 2018 年提出 [《MobileNetV2: Inverted Residuals and Linear Bottlenecks》](https://arxiv.org/abs/1801.04381)。高精度网络往往需要大量乘加和内存，难以直接部署在移动设备。MobileNetV2 用逐通道卷积、倒残差和线性瓶颈，在较小计算预算下保留有用的图像特征。这个设计后来也用于轻量目标检测和分割骨干。

## 2. 源码导航与直观解释

读 [mobilenet_v2.py](mobilenet_v2.py) 时先看 `InvertedResidual`：输入通道先用 1×1 卷积**扩张**，再由 `groups=hidden` 的 3×3 卷积在每个通道内处理空间图案，最后用 1×1 卷积**压缩**回较窄输出。只有步长为 1 且输入输出通道一样时才执行 `x+result`。最后的压缩层故意没有 ReLU。`MobileNetV2` 把七组此类块串起来，再进行全局平均池化和分类。输入 `[B,3,H,W]`，输出 `[B,类别数]` logits。

从仓库根目录运行 `python "MobileNetV2/mobilenet_v2.py"`。底部示例缩小通道；代码中的 `width_mult` 让你观察“模型更窄时参数和计算如何变化”。

## 3. 数学解释

普通 `K×K` 卷积的乘加量大约是 `H·W·C_in·C_out·K²`。先做逐通道卷积只需约 `H·W·C_in·K²`，再做 1×1 逐点卷积约需 `H·W·C_in·C_out`；源码 `groups=hidden` 令每个输入通道只与自己的卷积核连接。倒残差块可概括为 `y=x+P(D(E(x)))`（形状相同时），其中 `E` 扩张、`D` 逐通道处理、`P` 线性压缩。若在窄层压缩后再加 ReLU，很多信息可能直接变成零，因此 `P` 后没有激活。这里的“倒”指残差连接跨的是窄通道层，而主要计算发生在扩张后的宽通道层。
