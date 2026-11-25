# SegNet：用最大池化下标恢复像素位置

## 1. 出处、提出者与意义

Vijay Badrinarayanan、Alex Kendall、Roberto Cipolla 在 2015 年提出 [《SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation》](https://arxiv.org/abs/1511.00561)。编码器的最大池化会压缩特征图，但每个池化窗口里“最大值原来在哪”仍是有用的位置线索。SegNet 在编码时保存这些下标，解码时把低分辨率特征放回对应位置，再用卷积填充空隙。它用较少的特征传递信息，体现了与 U-Net 拼接跳跃特征不同的恢复路线。

## 2. 源码导航与直观解释

[segnet.py](segnet.py) 的 `F.max_pool2d(..., return_indices=True)` 同时返回池化值和最大值下标。编码器共两次池化；解码器按相反顺序用 `F.max_unpool2d` 和对应下标展开网格，然后经过卷积与批归一化生成更密的特征。输入 `[B,3,32,32]`，输出 `[B,C,32,32]` 逐像素 logits。根目录运行 `python SegNet/segnet.py`。

原论文编码器拓扑对应 VGG16 的 13 个卷积层，解码器也更深。本目录保留最关键的索引池化与反池化，仅用两级小网络演示；没有预训练、道路场景数据或论文指标。

## 3. 数学解释

对每个 `2×2` 窗口 `R`，最大池化产生 `y=max_{i∈R}x_i`，同时记录 `a=argmax_{i∈R}x_i`。反池化 `Unpool(y,a)` 把 `y` 放回索引 `a`，窗口其余位置先置零；它并不能精确恢复原来的四个值，只恢复“强响应在哪”。源码的 `index2` 必须先给内层解码使用，`index1` 再恢复外层尺度；把顺序颠倒会导致通道或尺寸不匹配。反池化后的稀疏特征经可学习卷积扩散信息，最后用像素交叉熵训练分割类别。
