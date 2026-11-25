# TCN：因果膨胀卷积处理时间序列

## 1. 出处、提出者与意义

Shaojie Bai、J. Zico Kolter、Vladlen Koltun 在 2018 年 [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) 中系统比较了通用时序卷积网络与循环网络。TCN 用因果卷积避免读取未来，并通过逐层增大的 dilation 扩大感受野；残差块帮助堆叠更多层。

## 2. 源码导航与直观解释

[tcn.py](tcn.py) 的 `CausalConv1d` 只在左侧补零，使位置 `t` 的输出不能看到未来；`TemporalBlock` 包含两次这样的卷积和残差相加。`TCN` 按 `1,2,4,…` 设置 dilation，输入 `[B,T,D]`，输出 `[B,T,O]`。运行 `python TCN/tcn.py`。示例省去论文常用的 weight norm 与 dropout，但因果性、膨胀卷积和残差骨架均在源码中。

## 3. 数学解释

一维膨胀卷积可写成 `y_t=Σ_{i=0}^{k−1} w_i x_{t−i·d}`，其中 `d` 是 dilation；索引只取 `≤t`，所以严格因果。`k=3` 时，`d=1` 观察近邻，`d=4` 可直接访问更早的时间点。每个块输出 `h_{l+1}=h_l+F_l(h_l)`；层数增加时，多个膨胀尺度叠加扩大可见历史。源码用 `F.pad(x,(left_padding,0))` 保证输出时间长度不变。

对于核宽 3、dilation=4 的单次卷积，输出 y_t 读取 x_t、x_{t-4}、x_{t-8}；在 t<8 时缺失的更早位置由左侧零填充。测试脚本只改最后一个输入并比较先前所有输出，确认未来不会泄露。两层卷积串联后，块的有效感受野比单层更大，但输出长度保持 T 不变。
