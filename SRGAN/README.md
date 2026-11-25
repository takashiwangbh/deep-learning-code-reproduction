# SRGAN：让超分辨率图像有更真实的纹理

## 1. 出处、提出者与意义

Christian Ledig、Lucas Theis、Ferenc Huszár、Jose Caballero 等人在 2016 年提出、CVPR 2017 发表 [《Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network》](https://arxiv.org/abs/1609.04802)。单纯最小化像素误差常得到平滑的高分辨率图；SRGAN 将残差超分网络、感知内容损失和对抗学习结合，追求肉眼看来可信的细纹理。它是理解“像素最接近”与“视觉上真实”并不完全相同的经典案例。

## 2. 源码导航与直观解释

在 [srgan.py](srgan.py) 先看 `SRGenerator`：`ResidualBlock` 在低分辨率特征上处理内容，两次 `PixelShuffle(2)` 各把宽高翻倍，最终把 `[B,3,8,8]` 变成 `[B,3,32,32]`。`SRDiscriminator` 判断高分辨率图真假；`losses` 结合真假损失和特征空间内容损失。`FeatureExtractor` 的 VGG 风格层结构在本目录可读，但示例**没有预训练权重**：随机特征只用于验证梯度路径，不能当作论文的感知损失效果。真正训练需载入固定的、已训练好的特征网络权重以及高低分辨率成对图片。运行 `python SRGAN/srgan.py`。

## 3. 数学解释

令低分辨率输入为 `x_lr`、真高分辨率图为 `x_hr`，生成图 `x̂=G(x_lr)`。上采样子像素层先用卷积生成 `4C` 个通道，再由 `PixelShuffle(2)` 重排为 `C` 个通道和双倍宽高；重复两次就是 4×。内容损失可写为 `‖φ(x̂)−φ(x_hr)‖²`，`φ` 是固定的感知特征提取器；代码 `F.mse_loss(feature_extractor(fake), feature_extractor(high_resolution))` 对应它。生成器再加 `λ·softplus(−D(x̂))`，让生成图更像自然高分辨率图；判别器最小化真假二分类损失。若 `φ` 未经过训练，它只是随机投影，公式能运行但不保证感知质量，所以本项目没有声称复现论文图片效果。
