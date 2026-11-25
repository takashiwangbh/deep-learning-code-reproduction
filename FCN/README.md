# FCN-8s：把分类网络改成逐像素预测

## 1. 出处、提出者与意义

Jonathan Long、Evan Shelhamer、Trevor Darrell 于 2014 年提出 [《Fully Convolutional Networks for Semantic Segmentation》](https://arxiv.org/abs/1411.4038)。传统图像分类网络最后输出一张图的类别，无法直接指出每个像素是什么。FCN 把后面的全连接分类改成卷积分类，输出带空间位置的类别分数，再上采样回原图。FCN-8s 进一步融合深层语义与较浅层细节，是语义分割的重要起点。

## 2. 源码导航与直观解释

[fcn.py](fcn.py) 的 `conv_stage` 构建五个卷积与池化阶段，`FCN8s.forward` 保存 pool3、pool4、pool5 的特征。最深的 pool5 先产生粗类别图，上采样到 pool4 后与 pool4 的类别图相加；再上采样到 pool3 与 pool3 相加；最后恢复输入宽高。输入 `[B,3,H,W]`，输出 `[B,C,H,W]`，每个像素的 `C` 个值都是未归一化 logits。根目录运行 `python FCN/fcn.py` 验证前向和反向传播。

教学模型通道数较小，上采样使用双线性插值；原论文把分类网的权重迁移过来，使用可学习反卷积等配置。代码不带预训练权重，也不包含数据集与分割指标。

## 3. 数学解释

若最深特征长宽只剩 `H/32×W/32`，在该网格上做 `1×1` 卷积相当于给每个粗位置做独立分类，得分 `S₅`。FCN-8s 逐级融合：`S₄=Up₂(S₅)+Score(pool4)`，`S₃=Up₂(S₄)+Score(pool3)`，最后 `Y=Up₈(S₃)`。其中 `Up₂` 和 `Up₈` 都让空间网格变大，源码用目标特征的精确尺寸插值，故奇数边长也能对齐。输出每个像素 `(h,w)` 可用交叉熵 `−log softmax(Y_{h,w})_{label}` 训练；分类类别在所有位置共享卷积参数，因而模型可处理不同图片尺寸。
