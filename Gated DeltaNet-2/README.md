# Gated DeltaNet-2（独立擦除与写入）

## 1. 出处、提出者与意义

Ali Hatamizadeh、Yejin Choi 和 Jan Kautz 在 [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention](https://arxiv.org/abs/2605.22791)（2026）中进一步拆开门控 Delta 的两个决定：旧记忆的哪些 key 通道该被擦除，及新 value 的哪些通道该被写入。上一代规则用同一个标量 `β` 同时控制两者；这篇工作让擦除门 `b` 和写入门 `w` 各自按通道变化。

阅读顺序建议先看 Gated DeltaNet 项目中的标量门，再看本项目的逐通道门。这里重现递推数学式与小型语言模型，未实现论文的分块 CUDA 算法、短因果卷积、分组 value head 和滑动窗口混合层。

## 2. 源码导航与直观解释

[gated_delta2.py](gated_delta2.py) 中的 `gated_delta2_scan` 逐时间步维护关联矩阵；`GatedDelta2Mixer` 从输入产生 `q/k/v`、衰减 `alpha`、擦除门 `erase` 和写入门 `write`；`GatedDelta2LanguageModel` 把这些层堆叠成 token 预测模型。RMSNorm 与 SwiGLU 的源码保存在本目录的 [modern_decoder.py](modern_decoder.py)，项目可以独立运行。

从仓库根目录运行 `python 'Gated DeltaNet-2/gated_delta2.py'`。输入整数 token `[B,T]`，输出每个位置的词表分数 `[B,T,V]`。为让递推式清晰，Python 循环一行行展示状态变化；内部状态和 RMSNorm 统计用至少 fp32 计算，避免半精度长序列累积误差。

## 3. 数学解释

把记忆矩阵 `Sₜ` 的行看作 key 通道，列看作 value 通道。第一步按通道衰减：`S̄ₜ = Diag(αₜ)Sₜ₋₁`，其中 `Diag(α)` 是以向量 `α` 为对角线的矩阵。第二步只用擦除门选中的 key 部分读旧值：`rₜ = S̄ₜᵀ(bₜ⊙kₜ)`。第三步写入 `Sₜ = S̄ₜ + kₜ(wₜ⊙vₜ−rₜ)ᵀ`；最终 `oₜ=Sₜᵀqₜ`。`⊙` 表示逐通道相乘。源码中的 `decayed`、`old_value`、`new_value` 和 `state` 依次对应这四步。

若把每个通道的 `α` 设为相同标量，并令 `b=w=β`，这个式子退化为上一代 Gated DeltaNet。测试使用同一组随机 `q/k/v` 比较两种写法的输出。因此这篇论文的新增自由度不是更长的公式，而是让“清除旧关联”和“接纳新内容”可分别选择通道。
