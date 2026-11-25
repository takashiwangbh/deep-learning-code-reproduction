# ControlNet：给冻结的扩散主干添加空间控制

## 1. 出处、提出者与意义

Lvmin Zhang、Anyi Rao、Maneesh Agrawala 在 2023 年提出 [《Adding Conditional Control to Text-to-Image Diffusion Models》](https://arxiv.org/abs/2302.05543)。单靠文本往往难以准确指定姿态、边缘或深度图中的空间布局。ControlNet 冻结已有扩散模型主干，另外训练一条读取控制图的分支，并用零初始化卷积把控制信息接回主干。这样开始训练时输出与原模型一致，随后才逐渐学会利用新条件，是复用大型预训练生成模型的经典方案。

## 2. 源码导航与直观解释

[controlnet.py](controlnet.py) 的 `TinyDenoiser` 是小型时间条件噪声预测器。`ControlNet` 将传入的主干复制为 `locked` 并冻结参数；另复制其特征块为可训练支路。`condition_encoder` 提取控制图特征，`zero_condition` 负责注入支路，`zero_residual` 把支路输出接回冻结主干。两个零卷积的权重和偏置初始都为零，故初始化后控制图暂不改变主干预测。输入噪声图 `[B,C,H,W]`、时间 `[B]`、控制图 `[B,K,H,W]`，输出预测噪声 `[B,C,H,W]`。运行 `python ControlNet/controlnet.py`。本例只有一个控制注入点，省略文本、U-Net 多尺度层和预训练权重。

## 3. 数学解释

记冻结主干提取的特征为 `h_f(x,t)`，可训练副本为 `h_c(x,t,z)`，控制图为 `z`。小型示例的输出可以写成 `ε(x,t,z)=Head_f[h_f(x,t)+Z_out(h_c(x,t)+Z_in(E(z)))]`，`Z_in/Z_out` 为零初始化卷积。训练开始时 `Z_out=0`，所以 `ε(x,t,z)=Head_f[h_f(x,t)]`，与传入主干输出相同；随后 `Z_out` 学出非零权重，控制路径的梯度逐步进入更早的支路。代码中 `requires_grad_(False)` 冻结旧参数，但冻结层仍可把梯度传回新增残差。扩散训练继续用预测噪声与真实噪声的 MSE，只更新控制分支相关参数。
