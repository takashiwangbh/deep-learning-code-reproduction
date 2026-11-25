# Classifier-Free Guidance：同一模型的条件与无条件预测组合

## 1. 出处、提出者与意义

Jonathan Ho 和 Tim Salimans 在 2022 年提出 [《Classifier-Free Diffusion Guidance》](https://arxiv.org/abs/2207.12598)。想让扩散模型更贴近指定类别或文本，可以使用外部分类器给采样方向施加梯度；CFG 则让一个噪声预测模型既学“有条件”也学“无条件”的任务，采样时只需做两次模型预测并线性组合。条件越强，通常越贴合提示，但也可能牺牲多样性；它已成为条件扩散的重要机制。

## 2. 源码导航与直观解释

[cfg.py](cfg.py) 的 `ConditionalNoisePredictor` 有一个专门的“空条件”嵌入。`drop_conditions` 在训练中随机把真实类别换成空条件；`GuidedDiffusion.training_loss` 对加噪向量让网络预测所加噪声，混合训练两种任务。采样时 `guided_prediction` 对同一 `x_t` 分别调用无条件和有条件预测，再由 `scale` 组合；`GuidedDiffusion.sample` 用这个结果逐步去噪。输入为向量 `[B,D]`、时间 `[B]` 和类别 `[B]`；根目录运行 `python 'Classifier-Free Guidance/cfg.py'`。示例使用少量离散类别，不含文本编码器、图像 U-Net 或真实训练数据。

## 3. 数学解释

训练时以概率 `p` 丢弃条件 `c`，使同一个 `εθ(x_t,t,c)` 同时学会 `c` 与空条件 `∅` 的噪声预测。采样时先算 `ε_u=εθ(x_t,t,∅)`、`ε_c=εθ(x_t,t,c)`，再用 `ε_guided=ε_u+s(ε_c-ε_u)`；源码的 `scale=s`。`s=0` 等于无条件结果，`s=1` 等于普通条件结果，`s>1` 会沿两者差异进一步放大条件方向。噪声预测进入 DDPM 均值 `μ_t=[x_t-β_t ε_guided/√(1-ᾱ_t)]/√α_t`，从 `t=T` 倒推到 0。公式只规定方向；过大 `s` 可能使预测偏离训练分布。
