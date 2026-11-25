# Restricted Boltzmann Machine：可交替采样的能量模型

## 1. 出处、提出者与意义

Paul Smolensky 1986 年的 [Harmony Theory](https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Smolensky.1986.pdf) 讨论了这种只有可见层与隐藏层之间连接的受限玻尔兹曼结构。Geoffrey Hinton 2002 年的 [对比散度论文](https://pubmed.ncbi.nlm.nih.gov/12180402/) 让这类能量模型更容易训练。没有同层连接是关键：给定可见层时，隐藏单元相互独立，可以同时采样；反过来也一样。RBM 是早期无监督特征学习与 Deep Belief Network 逐层预训练的重要构件。

## 2. 源码导航与直观解释

[rbm.py](rbm.py) 的 `hidden_prob` 计算从二值可见向量 `[B,V]` 到隐藏激活概率 `[B,H]`，`visible_prob` 反向计算重建概率；`gibbs` 在两层之间交替做 Bernoulli 采样。`free_energy` 告诉模型一组可见状态有多“容易”出现，值越低相对越受偏好。`cd_loss` 从训练样本出发做 `k` 步 Gibbs 链，把数据相与模型生成的负相比较。运行 `python 'Restricted Boltzmann Machine/rbm.py'` 可见小批二值数据的前向和梯度。代码不宣称 `k=1` 就得到精确最大似然；真实训练还需长期采样、学习率和数据预处理选择。

## 3. 数学解释

可见位 `v_i`、隐藏位 `h_j` 的联合能量为 `E(v,h)=-aᵀv-bᵀh-vᵀWh`。因为同层之间没有边，条件分布分解为 `P(h_j=1|v)=sigmoid(b_j+Σ_i v_iW_ij)` 与 `P(v_i=1|h)=sigmoid(a_i+Σ_j W_ijh_j)`；这正是两个概率函数。把所有隐藏状态求和可得到可见自由能 `F(v)=-aᵀv-Σ_j softplus(b_j+vᵀW_:j)`。理想训练要让数据自由能下降、模型样本自由能上升；`cd_loss=mean F(data)-mean F(Gibbs_k(data))` 以有限步负样本近似这一梯度。负样本采样在 `no_grad` 下进行，避免误把离散采样当作可微路径。
