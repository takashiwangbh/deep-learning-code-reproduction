# Swin Transformer：交替使用普通窗口和移位窗口

## 1. 出处、提出者与意义

Ze Liu、Yutong Lin、Yue Cao 等人于 2021 年提出 [《Swin Transformer: Hierarchical Vision Transformer using Shifted Windows》](https://arxiv.org/abs/2103.14030)。整图自注意力在高分辨率图像上代价很高，也缺少视觉任务常用的多尺度特征。Swin 将注意力限制在小窗口，再让下一块窗口平移，使原先属于不同窗口的邻域能够通信；阶段间的 patch merging 逐渐降低宽高、提高通道数。它因此可作为分类、检测、分割的分层主干。

## 2. 源码导航与直观解释

在 [swin.py](swin.py) 中先读 `partition` 和 `reverse`：它们把 `[B,H,W,C]` 排成多个局部窗口再拼回图像。`WindowAttention` 只在窗口内做 Q/K/V；`SwinBlock` 在两层间交替使用普通窗口与平移窗口。`shifted_mask` 阻止 `torch.roll` 带来的左右、上下边缘错误相连；`PatchMerging` 将 2×2 邻域拼接后压成两倍通道。`SwinTransformer` 从图像 `[B,3,32,32]` 产生 `[B,C]` 分类 logits。根目录运行 `python 'Swin Transformer/swin.py'`。

这里是两阶段的小模型，窗口机制和合并机制都在本地源码中；第二阶段只有一个 4×4 窗口时会自动关闭无意义的平移。原论文更深、更多阶段，并包含相对位置偏置、训练配方及专用高效内核。默认尺寸仅用于 CPU 验证。

## 3. 数学解释

若每个窗口有 `M×M` 个 patch，一张图有 `HW/M²` 个窗口；每窗口注意力约处理 `M⁴` 对 token，因此总配对量约 `HW·M²`，固定 `M` 时随图像面积线性增长。普通窗口会让窗口之间无法交流：将特征图循环平移 `s=M/2` 后，新的窗口覆盖原窗口边界，再计算 `softmax(QKᵀ/√d)V`。循环平移会把图左端卷到右端；`shifted_mask` 对这些虚假的邻居设掩码。合并层把四个相邻向量 `[x₀₀,x₁₀,x₀₁,x₁₁]∈R^{4C}` 投影到 `R^{2C}`，因此空间宽高各减半，特征通道加倍。
