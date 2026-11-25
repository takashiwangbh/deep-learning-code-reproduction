# Longformer：局部窗口加少量全局注意力

## 1. 出处、提出者与意义

Iz Beltagy、Matthew E. Peters、Arman Cohan 于 2020 年提出 [《Longformer: The Long-Document Transformer》](https://arxiv.org/abs/2004.05150)。标准自注意力让长度 `T` 的每个词都看全部词，注意力矩阵需要 `T²` 个位置；长文档很快耗尽内存。Longformer 让普通 token 主要看附近窗口，另给任务相关的少数 token 全局视野，例如文档分类时的首 token。局部传播保留上下文，全局位置负责汇总或远距离交换信息。

## 2. 源码导航与直观解释

[longformer.py](longformer.py) 的 `LocalGlobalAttention.forward` 为每个 query 构造真正要看的 key 下标：普通 query 选择左右 `window` 个邻居并加入所有全局 token；全局 query 可以读取全文。代码只对这些下标做 Q/K 点积，不分配完整 `T×T` 注意力矩阵。`Longformer` 再加入嵌入、残差、归一化和前馈层。输入 token ID 为 `[B,T]`，`global_mask [T]` 标记全局位置，输出 `[B,T,V]`。根目录运行 `python Longformer/longformer.py`。

这是单层、同批样本共用全局位置的教学版本。原论文还有专门的全局 Q/K/V 投影、扩张窗口、预训练与长文档任务设置；长序列高效运行需专门的向量化或内核，本例使用清楚的 Python 循环。

## 3. 数学解释

普通位置 `i` 的可见集合是 `N(i)={j:|j−i|≤w}∪G`，`G` 为全局 token 集；若 `i∈G`，则 `N(i)` 是所有位置。该行只在 `j∈N(i)` 上计算 `a_{ij}=softmax_j(q_i·k_j/√d)`，然后 `o_i=Σ_{j∈N(i)} a_{ij}v_j`。因此普通位置每层处理约 `2w+1+|G|` 个邻居；若全局位置很少，注意力计算量随 `T` 近似线性增长。源码的 `index` 就是 `N(i)`；可把全局位置理解成所有局部窗口都能找到的“公共公告栏”。
