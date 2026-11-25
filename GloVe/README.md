# GloVe：拟合全局词共现统计

## 1. 出处、提出者与意义

Jeffrey Pennington、Richard Socher、Christopher D. Manning 在 2014 年 [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/) 中提出用词与上下文的共现次数学习词向量。它与 Word2Vec 的局部预测目标不同，直接让两个词向量的内积解释语料统计。论文还强调共现概率比值与词义关系之间的联系。

## 2. 源码导航与直观解释

[glove.py](glove.py) 的 `GloVe` 为词和上下文各维护向量与偏置。`forward(word_ids,context_ids,count)` 只接受出现过、`count>0` 的共现对，计算加权平方误差；`word_vectors` 把两套向量相加作为常见的最终表示。三个输入形状均为 `[B]`，返回标量损失。运行 `python GloVe/glove.py`。本项目聚焦模型与目标函数，共现矩阵的分词和计数需另行准备。

## 3. 数学解释

若 `Xᵢⱼ` 是词 `i` 与上下文 `j` 的共现次数，GloVe 让 `wᵢᵀŵⱼ+bᵢ+b̃ⱼ≈log Xᵢⱼ`。损失为 `Σᵢⱼ f(Xᵢⱼ)(wᵢᵀŵⱼ+bᵢ+b̃ⱼ−log Xᵢⱼ)²`。示例的权重 `f(x)=min((x/x_max)^α,1)`，让极少见的配对不会主导训练，同时高频配对权重封顶。源码用 `count.log()` 与 `weighting` 明确对应这些项。

如果把某个共现次数加倍，目标 log X 只增加 log 2，而不是翻倍；这让大计数不会线性压倒小计数。权重函数对低频配对也较温和，高频则封顶。可以用一个很小的共现表手工算出单个配对损失，再和 forward 输出比较；注意未出现的 X=0 不进入训练，因为 log 0 未定义。
