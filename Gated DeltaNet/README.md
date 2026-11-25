# Gated DeltaNet（门控 Delta 网络）

## 1. 出处、提出者与意义

Songlin Yang、Jan Kautz 和 Ali Hatamizadeh 在 [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)（ICLR 2025）中提出门控 Delta 规则。一般的状态衰减会同时淡化所有旧信息；Delta 规则则能针对当前 key 修改特定记忆。论文把两种操作结合：先决定保留多少旧状态，再有针对性地修正一个 key 对应的 value。

这是一种固定大小记忆的序列模型思路。项目保留论文最核心的递推，并用小模型展示它如何与注意力层交替堆叠；没有复现论文的高性能分块扫描、短卷积、滑动窗口注意力或论文规模训练。

## 2. 源码导航与直观解释

先读 [gated_delta.py](gated_delta.py) 的 `gated_delta_scan`：`state` 是一个 key 到 value 的小型关联表，时间循环中每次先遗忘一部分，再对当前 key 覆写内容。`GatedDeltaMixer` 从输入生成 query、key、value 与两个门；`HybridDeltaAttentionLM` 交替使用递推层和普通因果注意力层。所需的 RMSNorm、SwiGLU 与 GQA 基础层源码也放在本项目的 [modern_decoder.py](modern_decoder.py) 中，运行不依赖另一个项目文件夹。

输入 token ID `[B,T]`，输出 `logits:[B,T,V]`；其中 `B` 是批大小、`T` 是长度、`V` 是词表大小。根目录运行 `python 'Gated DeltaNet/gated_delta.py'`，会对随机序列执行一次前向和反向传播。学习时可先只看 `gated_delta_scan` 与它的张量形状，再看外围语言模型。

## 3. 数学解释

让 `Sₜ` 表示第 `t` 步的记忆矩阵，`kₜ` 是当前 key，`vₜ` 是要写入的 value，`qₜ` 是读取记忆的 query。论文规则写成 `Sₜ = αₜ Sₜ₋₁ (I−βₜ kₜkₜᵀ) + βₜ vₜkₜᵀ`，输出为 `oₜ = Sₜqₜ`。`αₜ` 在 0 到 1 之间，控制整体保留多少旧状态；`βₜ` 控制本次定向更新的力度。`I` 是恒等矩阵。

把式子展开，更容易对照源码：先算旧记忆在当前 key 上读到的值 `old = Sₜ₋₁kₜ`；然后 `Sₜ = αₜSₜ₋₁ + [βₜ(vₜ−αₜold)]kₜᵀ`。方括号是应补写的差值，而非把 `vₜ` 无条件累加。若 `α=β=1` 且 key 为单位向量，第二次写同一 key 会替换第一次的值；本仓库的检查包含这一例子。源码在读取输出后再做逐 head 归一化与 SiLU 门控，以免递推输出尺度过大。
