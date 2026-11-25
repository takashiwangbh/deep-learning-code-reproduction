# 01 · 从梯度到训练循环

这里从最小的网络开始。所有示例不下载数据、不调用预训练模型；`mlp_numpy.py` 和 `optimizers_numpy.py` 仅依赖 NumPy。

| 文件 | 先看懂什么 | 核心公式 |
| --- | --- | --- |
| `mlp_numpy.py` | 前向、交叉熵、链式法则、XOR | `h=tanh(XW₁+b₁)`；`p=softmax(hW₂+b₂)` |
| `optimizers_numpy.py` | SGD、动量、Adam 和偏差校正 | `θ←θ−η·m̂/(√v̂+ε)` |
| `normalization.py` | BatchNorm 与 LayerNorm 的统计维度 | `(x−μ)/√(σ²+ε)·γ+β` |
| `train_mlp.py` | DataLoader、训练/验证、梯度清零 | `zero_grad → forward → loss → backward → step` |

从仓库根目录先运行：

```bash
python Deep-Learning-Sourcebook/foundations/mlp_numpy.py
python Deep-Learning-Sourcebook/foundations/optimizers_numpy.py
python Deep-Learning-Sourcebook/foundations/train_mlp.py
```

如果系统 Python 没有 PyTorch，请先按上级 README 安装依赖。`train_mlp.py` 会在自动生成的圆形分类数据上训练，输出验证准确率；它说明训练循环，并不代表真实数据集的基准成绩。

## 从数学到代码

在 `mlp_numpy.py` 中，`dlogits = (probabilities - one_hot(y)) / N` 是 softmax 与交叉熵合并后的导数；`dw2 = a1.T @ dlogits` 是矩阵乘法对权重的导数；`dz1 = da1 * (1-a1**2)` 是经过 `tanh` 的链式法则。每个梯度的形状都与对应参数一致。

BatchNorm 对同一特征在 batch 维度上计算均值；LayerNorm 对同一样本的特征维度计算均值。因此推理阶段 BatchNorm 需要保存训练统计量，而 LayerNorm 不需要。

## 原始论文

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Batch Normalization](https://arxiv.org/abs/1502.03167)
- [Layer Normalization](https://arxiv.org/abs/1607.06450)
