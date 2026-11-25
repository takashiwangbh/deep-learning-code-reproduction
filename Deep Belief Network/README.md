# Deep Belief Network：逐层预训练深层生成模型

## 1. 出处、提出者与意义

Geoffrey E. Hinton、Simon Osindero、Yee-Whye Teh 在 2006 年的 [《A Fast Learning Algorithm for Deep Belief Nets》](https://www.stats.ox.ac.uk/~teh/research/unsup/nc2006.pdf) 提出了以受限玻尔兹曼机（RBM）逐层贪心训练深层概率网络的方法。当时深层网络直接训练很困难，先让第一层学输入，再让第二层学第一层的表示，为后来的深度学习训练路线提供了重要突破。DBN 的上面两层构成无向的联想记忆，下面各层按方向从高层生成低层。

## 2. 源码导航与直观解释

[dbn.py](dbn.py) 内完整定义了本项目自己的 `BernoulliRBM`，没有跨项目导入。`DeepBeliefNetwork.greedy_pretrain` 先用二值数据 `[B,V]` 训练底层 RBM，把它采样出的隐藏状态当成下一层的“数据”，依次向上训练；`encode` 给出顶层概率表示；`generate` 在顶部 RBM 做 Gibbs 采样，再通过下层条件分布向下生成可见位。根目录运行 `python 'Deep Belief Network/dbn.py'`，展示 `[6,4,3]` 的小网络。这里覆盖贪心预训练和自顶向下采样，未实现论文的对比唤醒睡眠微调、真实数据训练或识别任务评估。

## 3. 数学解释

第 `l` 个 RBM 先学习 `P_l(h^l|h^{l-1})` 与 `P_l(h^{l-1}|h^l)`，其中底层 `h^0` 就是输入。训练时下一层看到的是前一层采样得到的隐藏位，不是直接重新看原图；所以每一层可用同一个局部 CD-k 规则独立训练。RBM 的条件概率为 `P(h_j^l=1|h^{l-1})=sigmoid(b_j^l+Σ_i h_i^{l-1}W_ij^l)`。顶部两层保留无向 RBM 的 Gibbs 更新；往下则使用 `P(h^{l-1}|h^l)` 依次采样。`greedy_pretrain` 的循环和 `generate` 的反向循环，分别把这两个方向明确写在源码里。有限步 CD 与未微调参数只是教学近似，并非论文报告的精确生成质量。
