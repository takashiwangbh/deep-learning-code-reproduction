# Mask R-CNN：在检测框内预测每个实例的掩码

## 1. 出处、提出者与意义

Kaiming He、Georgia Gkioxari、Piotr Dollár、Ross Girshick 于 2017 年提出 [《Mask R-CNN》](https://arxiv.org/abs/1703.06870)。语义分割给每个像素一个类别，但同类的两个对象仍可能连在一起；实例分割需要区分每个对象。Mask R-CNN 在 Faster R-CNN 的候选框分类与回归之外，增加并行的逐像素掩码分支。论文特别指出 RoI Pool 把连续坐标取整会破坏像素对齐，因而引入双线性采样的 RoIAlign。

## 2. 源码导航与直观解释

本目录的 [faster_rcnn.py](faster_rcnn.py) 保留完整本地区域提议、NMS、候选框分类和定位源码；[mask_rcnn.py](mask_rcnn.py) 的 `MaskRCNN` 继承它，并用 `_align` 替换整数裁剪池化：在候选框内固定采样网格，`grid_sample` 双线性读取特征。`mask_head` 对每个候选区域输出 `[类别数+背景,6,6]` 掩码 logits；`training_loss` 将检测损失与正候选框的二元掩码损失相加。输入 `[B,3,H,W]`，结果是每图候选框、类别 logits、位置偏移和掩码 logits。根目录运行 `python 'Mask R-CNN/mask_rcnn.py'`。

教学版 RoIAlign 每个输出格只取一个中心采样点，原实现可用更多采样点与更高分辨率掩码；主干、FPN、数据增强和最终实例筛选也简化了。训练时把真实框加到候选集合，保证随机模型早期掩码分支有正样本。

## 3. 数学解释

RoI Pool 会把连续框坐标量化到整数特征位置，再做最大池化；量化误差对粗分类影响可能小，却会移动掩码边界。RoIAlign 在输出网格的连续坐标 `(x,y)` 上做双线性插值：`f(x,y)=Σ_{i,j∈四邻点}w_{ij}(x,y)f_{ij}`，权重由采样点到四邻像素的距离确定。源码构造每个 RoI 的 `grid` 并交给 `F.grid_sample`，无需 `torchvision.ops.roi_align`。总损失为 `L=L_RPN+L_cls+L_box+L_mask`；`L_mask` 只对匹配到真实目标的正 RoI 计算，并在其真实类别对应的掩码通道上做逐像素二元交叉熵，因此同类多个实例仍各自拥有独立掩码。
