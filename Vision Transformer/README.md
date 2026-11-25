# Vision Transformer：把图像切块当作序列

## 1. 出处、提出者与意义

Alexey Dosovitskiy、Lucas Beyer、Alexander Kolesnikov 等研究者在 2020 年提出 [《An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale》](https://arxiv.org/abs/2010.11929)。此前图像识别主要依赖卷积；ViT 研究能否把一张图拆成多个小块，像处理词序列那样让各块通过注意力交流。它说明在合适的预训练条件下，标准 Transformer 编码器也可以成为图像分类骨干，并影响后来大量视觉和多模态模型。

## 2. 源码导航与直观解释

在 [vit.py](vit.py) 中，`VisionTransformer.patch_embed` 用 `kernel_size=patch_size、stride=patch_size` 一次切块并投影；`cls_token` 是额外加在序列前面的“汇总位置”；`pos_embed` 告诉模型每个块原来在图中的位置。`EncoderBlock` 交替运行自注意力与 MLP，`MultiHeadSelfAttention` 则显式算 Q、K、V。输入固定尺寸 `[B,3,img_size,img_size]`，输出 `[B,类别数]` logits。

从仓库根目录运行 `python "Vision Transformer/vit.py"`。脚本用 32×32 小图、8×8 块演示；默认参数对应更大的 ViT-B/16 结构。随机权重与没有预训练的数据并不能表现论文性能。

## 3. 数学解释

若图像高宽都是 `H`、每块边长为 `P`，块数为 `N=(H/P)²`。每个块先投影为 `D` 维 token，加入 1 个 class token 后序列长度是 `N+1`。自注意力为 `Attention(Q,K,V)=softmax(QKᵀ/√d_head)V`，其中 Q/K 用来比较“这个块该看谁”，V 提供被读取的内容；源码 `MultiHeadSelfAttention.forward` 逐步实现该式。`EncoderBlock` 的残差式 `x←x+Attention(Norm(x))` 与 `x←x+MLP(Norm(x))` 让网络在保留已有表示的同时更新它。最后取第 0 个 class token 分类，它在多层注意力后已汇聚其他块的信息。
