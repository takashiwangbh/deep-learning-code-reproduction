# Temporal Fusion Transformer：可解释的多步预测

## 1. 出处、提出者与意义

Bryan Lim、Sercan O. Arik、Nicolas Loeff、Tomas Pfister 于 2019 年提出 [《Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting》](https://arxiv.org/abs/1912.09363)。真实预测常同时有静态属性、历史观测和未来已知的日历或计划信息；模型不仅要预测多个未来时刻，还应指出哪些变量与时间片段有影响。TFT 把变量选择、门控、局部循环层和时间注意力组合起来，是多步预测中影响很大的可解释基线。

## 2. 源码导航与直观解释

[tft.py](tft.py) 的 `VariableSelection` 为每个时刻各变量算 softmax 权重，并汇成一条表示；静态特征先生成上下文，让选择权重随样本改变。`TemporalFusionTransformer.forward` 先用历史 LSTM 读取 `[B,P,Vp]`，再以最后状态启动未来已知变量 `[B,F,Vf]` 的 LSTM。`GatedResidual` 决定保留局部信息还是接受新变换；未来时刻的单头时间注意力只读取所有历史和截至自身的未来位置，最后输出 `[B,F,Q]` 分位数预测。返回的变量选择权重与时间权重可直接观察。运行 `python 'Temporal Fusion Transformer/tft.py'`。本例保留上述核心通路，但静态协变量编码、变量选择子网及门控层比论文完整架构更小，没有真实数据处理和解释性评测。

## 3. 数学解释

对时间 `t` 的各变量 `x_{t,j}`，先由静态上下文 `c` 算 `a_{t,:}=softmax(Selection([x_t,c]))`，再得 `z_t=Σ_j a_{t,j} Embed_j(x_{t,j})`。每个 `a` 的和为 1，因此可以看见模型在该时刻偏重哪些变量。LSTM 对历史与未来已知输入做递推；门控残差 `LayerNorm(residual + sigmoid(g(x))⊙f(x))` 可以抑制不需要的复杂变换。时间注意力 `A=softmax(QKᵀ/√d + mask)` 中，mask 禁止未来第 `t` 步读取更晚的已知变量；它仍可读取全部过去。`quantile_loss` 使用分位数 `q` 的 pinball 误差 `max(q(y-ŷ),(q-1)(y-ŷ))`，让模型给出不同风险水平的预测，而不只是一条均值。
