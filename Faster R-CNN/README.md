# Faster R-CNN：先提候选区域，再分类与定位

## 1. 出处、提出者与意义

Shaoqing Ren、Kaiming He、Ross Girshick、Jian Sun 于 2015 年提出 [《Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks》](https://arxiv.org/abs/1506.01497)。当时两阶段检测器需要外部算法先猜“哪里可能有物体”，这一步慢且难与网络统一训练。Faster R-CNN 用共享图像特征上的区域提议网络 RPN 预测候选框，再由 RoI 分类与回归头完成检测。它把候选区域生成纳入神经网络，成为理解两阶段检测的重要基础。

## 2. 源码导航与直观解释

[faster_rcnn.py](faster_rcnn.py) 的 `make_anchors` 为特征图每个位置生成不同长宽比的参考框；`rpn_objectness` 和 `rpn_delta` 分别预测“有没有物体”与框偏移。`decode_boxes` 把偏移还原成候选框，`nms` 删除大量重叠框。`_roi_features` 将每个候选区域裁剪为固定 3×3 特征，`class_head` 与 `box_head` 再判断具体类别和更精细的偏移。`training_loss` 同时计算 RPN 和 RoI 阶段的分类、框回归损失。输入 `[B,3,H,W]`，输出每张图一份候选框与对应 logits。根目录运行 `python 'Faster R-CNN/faster_rcnn.py'`。

这是用小主干和 RoI 池化做的 CPU 教学版；训练时额外加入真实框确保 RoI 头收到正样本。空标注图片只训练背景分类，不计算框回归。原论文主干、采样比例、交替训练、多尺度和推理后处理更完整。这里只复现核心两阶段机制，不提供可用的预训练检测器。

## 3. 数学解释

每个 anchor `a=(a_x,a_y,a_w,a_h)` 对应四个偏移 `t`，解码成 `x=a_x+t_xa_w, y=a_y+t_ya_h, w=a_w exp(t_w), h=a_h exp(t_h)`；`encode_boxes` 给出训练目标的逆变换。RPN 用 `IoU=交集面积/并集面积` 判断正负框：本例 `IoU≥0.7` 为正、`<0.3` 为负，中间忽略；每个真实框的最佳 anchor 也被强制设为正。损失包括 RPN 物体性二元交叉熵与正框 Smooth L1。RoI 阶段把候选框匹配到真实框，分类使用含背景类的交叉熵，正框再做 Smooth L1。NMS 按分数从高到低保留候选，删去与已保留框重叠过大的框；它是推理筛选，不是可微损失。
