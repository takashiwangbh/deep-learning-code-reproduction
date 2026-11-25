# pix2pix：有配对数据的图像翻译

## 1. 出处、提出者与意义

Phillip Isola、Jun-Yan Zhu、Tinghui Zhou 与 Alexei A. Efros 在 2016 年提出、CVPR 2017 发表 [《Image-to-Image Translation with Conditional Adversarial Networks》](https://arxiv.org/abs/1611.07004)。它把“边缘图→照片”“标签图→街景”等任务统一为成对图像翻译：生成器看输入图，判别器看输入图与目标图的配对是否真实。U-Net 保留精细位置信息，PatchGAN 着重检查局部纹理，再用 L1 约束整体内容。

## 2. 源码导航与直观解释

在 [pix2pix.py](pix2pix.py) 先跟踪 `UNetGenerator.forward`：`down1/down2` 缩小特征，`up1/up2` 恢复尺寸，`torch.cat((u1,x1))` 把早期细节直接送到后段。`PatchDiscriminator` 把源图与目标图按通道拼接，输出 `[B,1,8,8]` 的**一张局部真假分数图**，而非整图一个分数。`losses` 计算判别器目标与生成器的对抗加 L1 目标。输入和生成图都是 `[B,3,32,32]`。运行 `python pix2pix/pix2pix.py`。这个小 U-Net 没有论文的大网络深度与 dropout，随机输入只检验结构。

## 3. 数学解释

给定成对的 `(x,y)`，生成器输出 `G(x)`；判别器判断 `(x,y)` 和 `(x,G(x))`。目标为条件 GAN 损失加 `λ‖y−G(x)‖₁`。代码的 `F.softplus` 实现真假二分类，`F.l1_loss` 对应第二项，默认 `λ=100`。对抗项鼓励像真实目标域的局部纹理；L1 项把输出拉向与输入配对的正确结构。训练 D 时调用 `fake.detach()`，避免 D 更新把 G 也改动；训练 G 时不 detach，使梯度能沿 `D(source,fake)` 回到 U-Net。`PatchDiscriminator` 对每个局部格子求损失，再取平均。
