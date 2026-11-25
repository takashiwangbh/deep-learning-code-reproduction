# PointNet：直接读取无序点云

## 1. 出处、提出者与意义

Charles R. Qi、Hao Su、Kaichun Mo、Leonidas J. Guibas 在 2016 年提出 [《PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation》](https://arxiv.org/abs/1612.00593)。一件三维物体可以表示成一组 `(x,y,z)` 点，但点没有像图像那样固定的行列，文件里调换点的顺序不该改变物体类别。PointNet 对每个点使用相同的 MLP，再用对顺序不敏感的最大池化汇成全局表示，是直接处理点集的关键起点。

## 2. 源码导航与直观解释

[pointnet.py](pointnet.py) 的 `InputTransform` 先查看整组点，预测一个接近单位矩阵的 `3×3` 变换，用来对齐输入；`PointNet.forward` 把变换作用到 `[B,N,3]` 点云。`shared_mlp` 逐点生成特征，`max(dim=1)` 从所有点提取全局特征，`classifier` 输出 `[B,C]`。从仓库根目录运行 `python PointNet/pointnet.py`，可检查形状和梯度。这里实现输入变换、共享逐点网络和全局分类；论文中的特征变换正则、点级分割头和大规模训练未包含。

## 3. 数学解释

设点集 `X={x₁,…,x_N}`，模型形式为 `f(X)=γ(max_i h(T(X)x_i))`。`T(X)` 是由整组点预测的对齐矩阵，`h` 是所有点共享的特征网络，`max_i` 在每个特征维上选最大值，`γ` 是分类头。无论怎样排列点的顺序，参与最大值的集合都一样，因而 `f` 不变。源码把 `T` 初始化为单位矩阵加一个从零开始学习的修正量，训练初期不致任意扭曲坐标。单纯最大池化主要保留每个通道最显著的响应，不能直接表达所有细粒度的点间关系，这也是后续点云模型继续发展的原因。
