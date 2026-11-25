# Multilayer Perceptron（多层感知机）

## 1. 出处、提出者与意义

这份项目以 David E. Rumelhart、Geoffrey E. Hinton 和 Ronald J. Williams 于 1986 年发表的 [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) 为主要学习起点。论文讲清了如何把输出误差沿网络反向传递，进而调整隐藏层的权重。多层感知机（MLP）本身是逐层连接的网络；反向传播让这些层能一起学习，而不只是人工指定中间特征。

一个直观例子是 XOR：两个输入相同时输出 0，不同时输出 1。单层线性分类器无法画出足够复杂的边界；增加带非线性激活的隐藏层后，网络可以把输入变换到更容易分类的表示。后面的 CNN、Transformer 虽然结构不同，依然依靠前向计算、损失、反向传播、参数更新这一训练链条。

## 2. 源码导航与直观解释

- [mlp_numpy.py](mlp_numpy.py)：只用 NumPy 写两层网络。先看 `TwoLayerMLP.forward`，再看 `loss_and_gradients` 中每个参数梯度如何手算，最后看 `train_xor` 怎样反复更新参数。
- [train_mlp.py](train_mlp.py)：同样的训练流程改用 PyTorch 自动微分，包含数据批次、训练/验证模式、`zero_grad → forward → loss → backward → step`。这里生成二维圆形分类数据，不下载数据集。

从仓库根目录运行 `python 'Multilayer Perceptron/mlp_numpy.py'` 和 `python 'Multilayer Perceptron/train_mlp.py'`。NumPy 例子的输入 `x` 为 `[样本数, 2]`，输出 `logits` 为 `[样本数, 2]`，两列是两类的原始分数。PyTorch 例子也返回类别分数；验证时取分数最大的一类。随机初始化的数值会因环境略有变化，重点看损失下降与预测是否正确。

## 3. 数学解释

第一层计算 `z₁ = XW₁ + b₁`，随后 `h = tanh(z₁)`；第二层计算 `z₂ = hW₂ + b₂`。若一批有 `N` 个样本、输入维度 `D`、隐藏宽度 `H`、类别数 `C`，则 `X:[N,D]`、`W₁:[D,H]`、`W₂:[H,C]`，所以最终 `z₂:[N,C]`。`tanh` 是关键：若没有它，两次矩阵乘法可以合并为一次，网络仍只能表达线性关系。

`softmax(z₂)` 把每类分数转成概率 `p`；平均交叉熵为 `L = −(1/N)Σᵢ log pᵢ,正确类别`。把 softmax 和交叉熵一起求导，可得到源码中的 `dlogits = (p − one_hot(y))/N`。再按链式法则向前一层传：`∂L/∂W₂ = hᵀ dlogits`、`dh = dlogits W₂ᵀ`、`dz₁ = dh ⊙ (1−h²)`、`∂L/∂W₁ = Xᵀ dz₁`。`⊙` 表示逐元素相乘。读源码时可以逐项核对梯度形状是否与对应权重相同；`checks/test_foundations.py` 还用数值微分验证了这些手写导数。
