# Feature Pyramid Network：为不同大小目标提供强语义特征

## 1. 出处、提出者与意义

Tsung-Yi Lin、Piotr Dollár、Ross Girshick、Kaiming He、Bharath Hariharan、Serge Belongie 在 2016 年提出 [《Feature Pyramid Networks for Object Detection》](https://arxiv.org/abs/1612.03144)。卷积主干本来就会产生多种分辨率的特征：浅层分辨率高，却缺少丰富语义；深层语义强，却空间粗糙。FPN 从高层向低层传递语义，并与同分辨率的侧向特征相加，使每一级都适合检测对应尺度的物体。它后来成为众多目标检测和实例分割方法的常见组成部分。

## 2. 源码导航与直观解释

[fpn.py](fpn.py) 的 `TinyBackbone` 生成主干 `C2,C3,C4,C5`；`FeaturePyramidNetwork.laterals` 用 `1×1` 卷积把四层通道统一，`forward` 从 C5 开始上采样并逐级加上 C4、C3、C2 的侧向特征，`smoothing` 用 `3×3` 卷积整理融合结果，最后生成更粗的 P6。运行 `python 'Feature Pyramid Network/fpn.py'` 可看到 `64×64` 输入对应 P2 到 P6 的空间大小依次为 `16,8,4,2,1`，并执行反向传播。这里实现 FPN 特征提取本身，不附带分类框头、锚框和检测损失；那些属于检测器项目。

## 3. 数学解释

设主干第 `l` 层特征为 `C_l`，其侧向投影 `L_l=Conv1×1(C_l)`；最深层 `M_5=L_5`，向下逐层递推 `M_l=L_l+Upsample(M_{l+1})`，其中上采样到与 `L_l` 相同的宽高。输出 `P_l=Conv3×3(M_l)`，`P_6` 再把 P5 下采样。源码中的循环从 `level=2` 到 `0` 正是从深到浅传播；`size=lateral[level].shape[-2:]` 确保奇数尺寸也能精确对齐。融合让 P2 保有浅层的细位置，又能收到深层语义。两路相加要求通道数相同，所以每级都先过 `1×1` 侧向卷积。
