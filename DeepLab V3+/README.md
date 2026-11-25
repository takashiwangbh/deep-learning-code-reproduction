# DeepLabv3+：多尺度空洞卷积与边界解码

## 1. 出处、提出者与意义

Liang-Chieh Chen、Yukun Zhu、George Papandreou、Florian Schroff、Hartwig Adam 于 2018 年提出 [《Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation》](https://arxiv.org/abs/1802.02611)。高层特征语义强但空间分辨率低，浅层特征边缘清楚却缺少全局语义。DeepLabv3+ 结合 ASPP（不同空洞率的并行卷积与全局池化）和一个轻量解码器：先从多个尺度理解目标，再用低层细节修正边界。空洞可分离卷积减少计算量。

## 2. 源码导航与直观解释

[deeplab_v3plus.py](deeplab_v3plus.py) 的 `SeparableConv` 先对每个通道独立做 3×3 空洞卷积，再用 1×1 卷积混合通道。`ASPP` 并行执行普通 1×1、三种空洞率和全局平均池化，最后拼接融合。`DeepLabV3Plus` 从浅层 `low` 获取宽高约为原图 1/4 的细节，从 `high` 获取约 1/16 的语义；ASPP 后上采样，与压缩后的低层特征拼接，经解码器回到原图。输入 `[B,3,H,W]`、输出 `[B,C,H,W]`。根目录运行 `python 'DeepLab V3+/deeplab_v3plus.py'`。

教学版用小型本地主干和较小空洞率，避免 CPU 示例过大；原论文使用更强主干、更大输出步长配置、数据增强与训练细节。本目录展示的是 ASPP、可分离空洞卷积和解码融合的源码。

## 3. 数学解释

普通 3×3 卷积覆盖紧邻像素；空洞率 `r` 时采样位置变成 `p+r·k`，`k∈{−1,0,1}²`，不增加核权重数便扩大感受野。ASPP 计算 `A=[Conv₁(x), Atrous_{r₁}(x), Atrous_{r₂}(x), Atrous_{r₃}(x), GlobalPool(x)]`，其中全局分支先池化到 1×1 再上采样回特征图尺寸。可分离卷积把普通卷积成本约 `9C_in C_out` 拆为 `9C_in+C_in C_out`。解码器的 `concat(Up(A), low)` 保留高层语义与浅层轮廓，而最终上采样使每个原图像素都有类别分数。
