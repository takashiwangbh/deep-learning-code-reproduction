# LoRA：低秩适配

## 1. 出处、提出者与意义

出处是 Edward J. Hu、Yelong Shen、Phillip Wallis、Zeyuan Allen-Zhu、Yuanzhi Li、Shean Wang、Lu Wang 和 Weizhu Chen 2021 年的 [《LoRA: Low-Rank Adaptation of Large Language Models》](https://arxiv.org/abs/2106.09685)。完整微调会更新模型里大量权重；LoRA 冻结原权重，只学习一个用两个小矩阵组成的改变量。这样可以用少得多的可训练参数适配新任务，也便于分别保存多个任务的适配器。

本项目用一个线性层讲清核心机制：复制已有层、冻结基础权重、训练低秩矩阵，以及推理时把它们合并回基础权重。论文还研究把 LoRA 用于大模型的多个位置；这里只实现基础 `nn.Linear` 适配器，便于看见每一步计算。

## 2. 源码导航与直观解释

打开 [lora.py](lora.py)，从 `LoRALinear.__init__` 看冻结的 `base` 和可训练的 `A/B`；`from_linear` 把已有 `nn.Linear` 的权重复制进来；`forward` 先算原层输出，再加低秩更新；`merge` / `unmerge` 在推理与继续训练之间切换。输入最后一维是 `in_features`，输出最后一维是 `out_features`，所以 `[B,T,D入]` 也能变成 `[B,T,D出]`。

想象原模型已学会一般语言，想让它适应新领域：保留原线性层不动，只调整旁边的小支路。初始化时 `B` 全零，小支路输出为零，适配器与原层完全相同；训练后小支路才逐步贡献变化。仓库根目录运行 `python LoRA/lora.py` 可看初始输出差值和可训练参数名称；[序列检查](../checks/test_sequence.py) 验证冻结权重、合并/撤销合并以及合并状态保存后的输出。

## 3. 数学解释

原线性层为 `y=Wx+b`，其中 `W∈R^(D出×D入)`。LoRA 改为 `y=Wx+b+(α/r)BAx`，其中 `A∈R^(r×D入)`、`B∈R^(D出×r)`、`r` 是选定的低秩宽度、`α/r` 是更新缩放。`LoRALinear.forward` 先用 `F.linear(x,A)` 得到 `r` 维，再用 `F.linear(…,B)` 回到 `D出` 维，因此不必显式构造完整的 `BA` 矩阵。

原权重有 `D出·D入` 个参数；LoRA 两个矩阵合计 `r(D入+D出)` 个可训练参数。当 `r` 远小于输入和输出宽度时，训练和保存的参数数目显著下降。`base.requires_grad_(False)` 冻结原层，`B=0` 让训练起点与原模型一致，随后梯度会逐渐更新小矩阵。

推理前可计算 `W′=W+(α/r)BA`，再直接用 `W′x+b`。`merge` 正是把更新加到 `base.weight`，`unmerge` 把它减回去；`_merged` 防止重复相加，并随状态字典保存。合并前后仅有浮点计算顺序带来的微小误差，数学上是同一个线性变换。
