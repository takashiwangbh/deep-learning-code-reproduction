# 视觉模型：从卷积到图像 Transformer

这里每个 `.py` 文件都包含可读的模型结构源码。只调用 PyTorch 的张量和基础层；不会从 `torchvision`、`timm` 或 `transformers` 导入现成模型。代码依据原论文独立编写，未复制第三方仓库。运行演示和测试不需要下载数据，也不代表模型已经训练好。

## 建议学习顺序

| 顺序 | 文件 | 原论文与年代 | 核心问题 |
| --- | --- | --- | --- |
| 1 | [`lenet5.py`](lenet5.py) | [LeNet-5, 1998](https://bottou.org/papers/lecun-98h) | 如何利用局部连接、共享权重和池化识别图像？ |
| 2 | [`alexnet.py`](alexnet.py) | [AlexNet, 2012](https://proceedings.neurips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | 较深 CNN 的 ReLU、LRN、dropout、分组卷积怎样协同？ |
| 3 | [`vgg16.py`](vgg16.py) | [VGG, 2014](https://arxiv.org/abs/1409.1556) | 为什么可以反复堆叠小 3×3 卷积？ |
| 4 | [`googlenet.py`](googlenet.py) | [GoogLeNet, 2014](https://arxiv.org/abs/1409.4842) | 如何并行地提取不同尺度特征？ |
| 5 | [仓库已有的 ResNet-34](../../Residual%20Network/ResNet-34%20.py) | [ResNet, 2015](https://arxiv.org/abs/1512.03385) | `F(x)+x` 如何让深层网络更容易优化？ |
| 6 | [`unet.py`](unet.py) | [U-Net, 2015](https://arxiv.org/abs/1505.04597) | 如何同时保留语义信息和像素位置？ |
| 7 | [`yolov1.py`](yolov1.py) | [YOLOv1, 2015](https://arxiv.org/abs/1506.02640) | 一次前向如何同时定位和分类物体？ |
| 8 | [`densenet121.py`](densenet121.py) | [DenseNet, 2016](https://arxiv.org/abs/1608.06993) | 如何复用此前所有层的特征？ |
| 9 | [`mobilenet_v2.py`](mobilenet_v2.py) | [MobileNetV2, 2018](https://arxiv.org/abs/1801.04381) | 如何降低移动端卷积计算量？ |
| 10 | [`simclr.py`](simclr.py) | [SimCLR, 2020](https://arxiv.org/abs/2002.05709) | 不用类别标签，如何让同一图像的两个视图更相似？ |
| 11 | [`vit.py`](vit.py) | [ViT, 2020](https://arxiv.org/abs/2010.11929) | 图像切块后能否直接用 Transformer 分类？ |

## 运行

从仓库根目录执行，需安装 Python 3.10+ 和 PyTorch：

```bash
python Deep-Learning-Sourcebook/vision/test_models.py
python Deep-Learning-Sourcebook/vision/vit.py
python Deep-Learning-Sourcebook/vision/yolov1.py
python Deep-Learning-Sourcebook/vision/simclr.py
```

第一条命令对所有模型生成小尺寸随机输入，检查输出形状、反向传播，并对 YOLOv1 的 IoU/责任框以及 SimCLR 的 NT-Xent 做可计算的公式检查。各文件底部的 `if __name__ == "__main__"` 也是独立可运行的最小示例。多数分类模型的默认构造参数对应论文中的模型规模；测试专门调小了宽度、分类头或输入尺寸来保证 CPU 可运行。

分类网络返回 **logits** `[B, 类别数]`，训练时可直接传给 `torch.nn.CrossEntropyLoss`，不要先做 softmax。U-Net 返回每个像素的 logits `[B, 类别数, H, W]`，标签为 `[B, H, W]`。GoogLeNet 在训练模式且启用辅助分类头时返回 `(主输出, 辅助1, 辅助2)`，通常对辅助损失赋较小权重；评估模式仅返回主输出。

YOLOv1 输出 `[N,S,S,C+5B]`：每格先有 `C` 个类别分数，再有 `B` 组 `(x,y,w,h,confidence)`。`YOLOv1Loss` 另接收物体掩码 `[N,S,S]`、真值框 `[N,S,S,4]` 和 one-hot 类别 `[N,S,S,C]`。`x,y` 是网格内的中心偏移，`w,h` 是整图比例；同一格最多表达一个物体。检测器未包含数据集标注转换和非极大值抑制，示例只演示模型与训练目标。

SimCLR 的 `make_two_views(images)` 对 `[N,3,H,W]` 的无标签图像独立增强两次；`SimCLR(view1,view2)` 返回两个 `[N,D]` 投影，再传给 `nt_xent_loss`。索引相同的两个视图为正例，同一批的其他 `2N−2` 个视图为负例。预训练后使用 `model.representation(images)` 得到投影头之前的特征 `h`。

## 把数学和代码对起来

- **LeNet-5**：无填充的 5×5 卷积将 `32×32 → 28×28`，2×2 平均池化再将 `28×28 → 14×14`。`Conv2d` 的同一个卷积核在所有位置复用，减少参数量。
- **Inception v1**：四条分支分别计算 1×1、3×3、5×5 和池化特征，最后沿通道维 `torch.cat(..., dim=1)`。3×3/5×5 前的 1×1 卷积先减少通道，降低计算成本。
- **DenseNet**：若输入已有 `C` 个通道、每层新产生 `k` 个通道，第 `l` 层后共有 `C + lk` 个通道；关键行在 `DenseBlock.forward` 的 `torch.cat`。
- **MobileNetV2**：常规卷积约需 `H·W·C_in·C_out·K²` 次乘加；深度卷积约需 `H·W·C_in·K²`，然后由 1×1 卷积混合通道。`groups=hidden` 就是逐通道卷积。最后一个 1×1 层不接激活函数，保持线性瓶颈。
- **U-Net**：`Up.forward` 把上采样后的低分辨率语义特征与编码器的高分辨率特征拼接；这样每个输出像素兼有类别线索和位置信息。
- **YOLOv1**：`box_iou` 将格内 `(x,y)` 换算到整图坐标。每个有物体的网格用当前 IoU 最大的框负责坐标回归；`YOLOv1Loss` 依次求 `5·[(Δx)²+(Δy)²+(Δ√w)²+(Δ√h)²] + 责任框置信度误差² + 0.5·非责任框置信度误差² + 类别误差²`。责任框置信度目标是它与真值框的 IoU，不是固定的 1；其余框的置信度目标为 0。
- **SimCLR**：`nt_xent_loss` 对 `2N` 个 L2 归一化投影构造余弦相似度矩阵。对锚点 `i`，`Lᵢ=logsumexp(sim(i,k)/τ, k≠i)−sim(i,positive(i))/τ`；分母排除自身，但包括对应正例，再对双向 `2N` 个锚点取均值。
- **ViT**：`Conv2d(kernel_size=P, stride=P)` 将每个 `P×P` 图像块投影到一个 token；注意力显式计算 `softmax(QKᵀ/√d_head)V`。第 0 个 class token 用于分类。

## 与原论文的可见差异

这是教学复现，模型核心结构写全，但完整训练配方、预训练权重和数据集并不包含在内。LeNet-5 的 C3 改为全通道连接、S 层改为固定平均池化；AlexNet 的原始双 GPU 运行改为单机分组卷积；GoogLeNet 的辅助头使用自适应池化，以兼容小尺寸练习输入；U-Net 使用 `padding=1` 保持输入输出像素尺寸一致，而原论文采用裁剪跳接。YOLOv1 只对预测宽高加 `softplus` 正值约束，论文最后一层完全线性；SimCLR 示例使用较小的 ResNet-18 风格编码器和简化增强，论文主要实验采用 ResNet-50、强增强及大批量训练。这些差异影响与论文原始权重、指标的逐项复现，阅读时请对照代码注释。
