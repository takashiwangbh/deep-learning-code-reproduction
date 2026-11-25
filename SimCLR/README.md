# SimCLR：不用类别标签学习图像特征

## 1. 出处、提出者与意义

Ting Chen、Simon Kornblith、Mohammad Norouzi 和 Geoffrey Hinton 在 2020 年发表 [《A Simple Framework for Contrastive Learning of Visual Representations》](https://arxiv.org/abs/2002.05709)。大量图像没有人工类别标签，但同一图像的不同裁剪与颜色变化仍应表达相近内容。SimCLR 用两次随机增强构造正样本对，让网络在表示空间中拉近它们、区分同批其他图像，从而进行自监督预训练。它突出强数据增强、非线性投影头和较大批量的重要性。

## 2. 源码导航与直观解释

源码 [simclr.py](simclr.py) 按 `make_two_views → SimCLR.forward → nt_xent_loss` 阅读。`augment_view` 对同批 `[N,3,H,W]` 图像独立裁剪、翻转、颜色扰动和模糊，两次调用得到 `view1/view2`；共享的 `SmallResNetEncoder` 产生特征 `h`，两层 `projector` 再产生对比学习用的向量 `z`。`SimCLR.forward` 返回两组 `[N,D]` 投影，索引相同的两个向量组成正对。预训练后用 `model.representation(images)` 取得投影头之前的 `h` 用于下游任务。

从仓库根目录运行 `python "SimCLR/simclr.py"`。示例采用较小 ResNet-18 风格编码器和简化增强；论文主要实验使用更大的骨干、增强和训练批量。随机输入上的损失只证明计算路径可运行。

## 3. 数学解释

对一批 `N` 张图做两次增强，会得到 `2N` 个投影。先做 `z←z/||z||₂`，则两个投影的点积就是余弦相似度。对锚点 `i` 和它的另一视图 `j`，损失为 `L_i=−log[exp(sim(i,j)/τ)/Σ_{k≠i}exp(sim(i,k)/τ)]`，其中 `τ` 是温度，控制模型对相似度差异的敏感程度。分母包含正例 `j`，排除锚点自己；其余 `2N−2` 个视图是批内负例。源码 `nt_xent_loss` 用矩阵乘法得到所有相似度，把对角线设为 `−∞` 排除自己，再用 `logsumexp` 稳定计算分母，并对双向全部锚点取平均。这样无需写任何类别标签。
