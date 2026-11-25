# BYOL：没有负样本的自监督学习

## 1. 出处、提出者与意义

Jean-Bastien Grill、Florian Strub、Florent Altché 等人在 2020 年提出 [《Bootstrap your own latent》](https://arxiv.org/abs/2006.07733)。通常对比学习既需要“同图的正对”，也需要“大量不同图的负对”；BYOL 展示了只让同一图像两种视图互相预测，也能学出有用表示。关键是在线网络有额外预测头，而目标网络是在线网络的缓慢移动平均。它让研究者重新思考自监督表示学习为何不会简单塌缩成常量。

## 2. 源码导航与直观解释

[byol.py](byol.py) 中在线分支为 `image_encoder → projector → predictor`，目标分支只有 `image_encoder → projector`。`BYOL.forward` 对两张增强图像分别走两路，在线 A 预测目标 B、在线 B 预测目标 A；目标表示用 `torch.no_grad()` 截断梯度。`update_target` 在在线优化器更新之后，以 EMA 更新目标参数，批归一化运行统计也同步。输入 `[B,3,H,W]`，返回一个标量损失；根目录运行 `python BYOL/byol.py`。小卷积网络与随机视图只是机制演示，不复现论文的增强方案和表示质量。

## 3. 数学解释

设在线编码与投影为 `z_θ(v)`，预测头为 `q_θ`，目标投影为 `z_ξ(v')`。单方向误差 `L(v,v') = 2 - 2·cos(q_θ(z_θ(v)), stopgrad(z_ξ(v')))`；总损失还加反向 `L(v',v)`。余弦相似度先按向量长度归一化，使目标主要比较方向。`stopgrad` 对应源码的 `no_grad` 和 `.detach()`，因此目标网络不会用普通梯度追着在线网络跑。目标参数另按 `ξ ← m ξ + (1-m)θ` 缓慢移动。源码中的两次对称计算和 `update_target` 分别对应这两步；仅靠这几行随机示例不能证明一定不会塌缩，真实训练还依赖架构、归一化和增强设计。
