# StyleGAN2：用风格调制每一层卷积

## 1. 出处、提出者与意义

Tero Karras、Samuli Laine、Miika Aittala、Janne Hellsten、Jaakko Lehtinen 与 Timo Aila 在 2019 年 [《Analyzing and Improving the Image Quality of StyleGAN》](https://arxiv.org/abs/1912.04958) 中提出通常称为 StyleGAN2 的改进。它重写生成器的归一化方式：不同的潜在风格直接调制卷积权重，再按输出通道解调，减少早期 StyleGAN 的图像伪影。论文还使用路径长度正则来改善潜空间和图像变化的关系。

## 2. 源码导航与直观解释

打开 [stylegan2.py](stylegan2.py)：`StyleGenerator.mapping` 把输入噪声 `z` 映射成中间风格 `w`；`ModulatedConv2d` 针对批次中每个样本生成不同卷积核，在 `synthesis` 中从可学习的 4×4 常量逐步生成 `[B,3,8,8]` 图片。`NoiseInjection` 提供局部随机变化，`StyleDiscriminator` 输出真伪分数。`r1_penalty` 约束判别器对真图的输入梯度，`path_length_penalty` 计算生成图对风格的敏感度。根目录运行 `python StyleGAN2/stylegan2.py`。这是能读清关键机制的 8×8 教学版，省略了论文高分辨率网络、滤波上采样、风格混合、移动平均等训练细节。

## 3. 数学解释

对第 `b` 个样本，风格仿射变换给每个输入通道一个系数 `s_{b,i}`；原卷积核 `W_{o,i,k}` 先调制为 `W'_{b,o,i,k}=s_{b,i}W_{o,i,k}`。再对每个输出通道解调：`W''_{b,o,i,k}=W'_{b,o,i,k}/√(Σ_{i,k}(W'_{b,o,i,k})²+ε)`。`ModulatedConv2d.forward` 用分组卷积把这组样本专属核一次算完；`to_rgb` 不解调。路径长度项先让图像乘随机方向并对 `w` 求梯度，再惩罚该梯度长度偏离目标值；源码演示固定目标值，正式训练通常用移动平均目标。R1 项是 `E_x‖∇_x D(x)‖²`。这些机制与普通 GAN 的 MLP 生成器有本质区别，但随机权重仍无法产生可信图片。
