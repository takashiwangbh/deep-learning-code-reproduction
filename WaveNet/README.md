# WaveNet：门控因果卷积逐点生成

## 1. 出处、提出者与意义

Aaron van den Oord 等人在 2016 年 [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) 中提出自回归原始音频模型。它用膨胀因果卷积快速覆盖长历史，并以门控激活和跳跃连接组合多层特征。每个位置预测下一个音频离散值的分布，生成时再一步一步采样。

## 2. 源码导航与直观解释

[wavenet.py](wavenet.py) 的 `GatedCausalBlock` 用左侧填充、dilation=1/2/4/… 的卷积得到 filter 与 gate 两支，乘合后输出残差和 skip。`WaveNet` 将各层 skip 求和，再映射为每位置 `[词表]` logits。输入离散 ID `[B,T]`，输出 `[B,T,V]`，第 `t` 行用于预测下一位置；运行 `python WaveNet/wavenet.py`。示例用小离散词表代替 8-bit µ-law 音频编码，未包含条件语音合成系统。

## 3. 数学解释

门控激活为 `z=tanh(W_f*x)⊙σ(W_g*x)`；一支提供候选特征，一支决定保留多少。膨胀因果卷积只读取 `x_t,x_{t−d},…`；随着 `d` 成倍增长，有限层数也能看到更长历史。自回归联合概率分解为 `p(x₁:T)=∏_t p(x_t|x_<t)`。训练时把真实前缀作为输入并并行计算各步 logits；真正生成时需把上一采样结果加入前缀。

第 t 个位置的卷积可以使用当前输入 token x_t，因此示例约定其 logits 用来预测 x_{t+1}，而不是重新预测 x_t。训练标签需要整体右移一位。检查脚本改动最后一个 token 后要求之前所有位置的 logits 不变，防止卷积填充方向弄反。真正生成音频时模型必须逐样本采样，速度与训练时不同。
