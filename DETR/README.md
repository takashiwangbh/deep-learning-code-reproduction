# DETR：用对象查询直接预测一组目标

## 1. 出处、提出者与意义

Nicolas Carion、Francisco Massa、Gabriel Synnaeve 等人于 2020 年提出 [《End-to-End Object Detection with Transformers》](https://arxiv.org/abs/2005.12872)。经典检测器通常需要 anchor、候选框匹配和 NMS 去重。DETR 把检测看成**集合预测**：少量可学习对象查询同时读图像特征，每个查询要么给出一个目标类别与框，要么给出“无目标”。训练时用二分匹配把预测集合与真实集合一对一对应，避免多个查询争同一个物体。

## 2. 源码导航与直观解释

[detr.py](detr.py) 的 `position_2d` 给二维特征 token 加位置编码；`Encoder` 让图像位置交流，`Decoder` 让可学习的 `queries` 互相交流并从图像读取信息。`DETR.forward` 输出固定 `Q` 个类别 logits `[B,Q,C+1]` 和归一化 `cx,cy,w,h` 框 `[B,Q,4]`。`optimal_assignment` 为小样本穷举最优一对一匹配；`set_loss` 再计算类别、L1 框误差和 GIoU。根目录运行 `python DETR/detr.py`。

教学版为了纯 Python 可读性，匹配器只支持最多 10 个查询、4 个真实对象；大规模训练应采用 Hungarian 算法。没有真实目标的图片只计算“无目标”分类损失，框损失为零。主干与 Transformer 也较小，不含原论文的多层训练设置或预训练权重。

## 3. 数学解释

每个查询 `q_i` 经交叉注意力读取编码图像 `M`：`Attention(q_i,M,M)=softmax(q_iK_Mᵀ/√d)V_M`。模型预测 `Q` 个候选，目标只有 `M≤Q` 个；匹配成本对每个预测—真实对加上 `−p_i(class_j)`、`5‖b_i−g_j‖₁`、`2(1−GIoU(b_i,g_j))`。`optimal_assignment` 找总成本最小且不重复使用查询的分配，未分配查询标为 `no-object`。损失对全部查询做类别交叉熵，对匹配查询做 L1 和 GIoU；`no-object` 类在源码中降低权重为 0.1。这里的 GIoU 在普通 IoU 外惩罚两框共同最小外接框中的空白，互不相交时仍给出定位信号。
