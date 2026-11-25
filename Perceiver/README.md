# Perceiver：让固定数量的潜变量读取大量输入

## 1. 出处、提出者与意义

Andrew Jaegle、Felix Gimeno、Andrew Brock 等人于 2021 年提出 [《Perceiver: General Perception with Iterative Attention》](https://arxiv.org/abs/2103.03206)。普通 Transformer 对全部输入 token 两两做注意力，在像素、音频等高维输入上太昂贵。Perceiver 使用一组较少、可学习的潜变量作为 query，通过交叉注意力读取大量输入；复杂计算主要发生在这组较短的潜变量内部。这样一个基本结构可以接收多种模态和不同长度的输入。

## 2. 源码导航与直观解释

[perceiver.py](perceiver.py) 的 `Perceiver.input_project` 先给 `inputs [B,N,C]` 拼接简化傅里叶位置特征，再映射到统一宽度。`latents [1,M,D]` 是网络自己学的“问题卡片”，其中 `M` 比 `N` 小。`cross` 让每张卡片读取所有输入，`self_attention` 让卡片互相交流；这个过程重复几次，最后对潜变量平均池化分类。根目录运行 `python Perceiver/perceiver.py`，默认示例输入 `[2,40,3]`、输出 `[2,10]`。

原论文对图像、音频和多模态输入使用不同位置编码与训练设置，并采用更完整的迭代块；此处用一维傅里叶坐标和小分类头展示关键瓶颈机制，不声称复现那些任务结果。

## 3. 数学解释

令输入 `X∈R^{N×D}`、可学习潜变量 `Z∈R^{M×D}`，`M≪N`。交叉注意力为 `Z'=Z+softmax((ZW_Q)(XW_K)ᵀ/√d)XW_V`：注意力权重只有 `M×N`，而不是输入之间的 `N×N`。随后潜变量自注意力使用 `M×M` 权重，前馈层逐潜变量变换。因此每轮核心注意力成本大约由 `MN+M²` 决定。`Perceiver.forward` 中 `cross(..., context)` 和 `self_attention(..., latent)` 分别对应两种注意力；反复读取输入使潜变量能够逐步整合更多信息。
