# N-BEATS：反向重建与前向预测的残差堆叠

## 1. 出处、提出者与意义

Boris N. Oreshkin、Dmitri Carpov、Nicolas Chapados、Yoshua Bengio 在 2019 年 [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://arxiv.org/abs/1905.10437) 中提出纯前馈的时间序列预测结构。每个块一边解释已经看到的历史，一边给未来贡献一部分预测；后续块只处理尚未解释的历史残差。这样无需循环或卷积也能构建深层预测模型。

## 2. 源码导航与直观解释

[nbeats.py](nbeats.py) 的 `NBeatsBlock` 将固定长度历史输入多层 MLP，再分成 `backcast`（重建历史）与 `forecast`（预测未来）两个线性头；`NBeats.forward` 逐块减去 backcast，并累加 forecast。输入 `[B,lookback]`，输出预测 `[B,horizon]` 和剩余历史残差 `[B,lookback]`。运行 `python 'N-BEATS/nbeats.py'`。这里是论文的 generic stack 思路，未加入可解释配置中的显式趋势/季节性基函数。

## 3. 数学解释

初始残差 `r₀=x`，初始预测 `ŷ₀=0`。第 `k` 块输出 `(b_k,f_k)=Block_k(r_k)`，随后 `r_{k+1}=r_k−b_k`、`ŷ_{k+1}=ŷ_k+f_k`。`b_k` 解释已经看到的窗口，`f_k` 为未来增添一部分；因此多个块像逐步修正同一个预测。代码里的 `residual= residual-backcast` 和 `forecast=forecast+addition` 正是这两条式子。

假设第一块的 backcast 恰好等于输入历史，则后续块收到的残差为零；但后续块仍可因偏置产生贡献，训练会共同调节所有块。generic 版本的 backcast 和 forecast 都是自由学习的线性头；可解释版本则把它们限制在趋势与季节性基函数上。这个区别在理解论文两种配置时很重要。
