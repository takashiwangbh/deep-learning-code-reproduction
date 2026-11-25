# Word2Vec：用上下文学习词向量

## 1. 出处、提出者与意义

Tomas Mikolov 等人在 2013 年的 [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) 提出高效的词向量训练框架；后续 [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) 详细介绍了负采样。这里实现 Skip-gram with Negative Sampling（SGNS）：给定中心词，分辨真实上下文词和抽来的噪声词。训练后，语境相近的词会获得可比较的向量。

## 2. 源码导航与直观解释

[word2vec.py](word2vec.py) 的 `SkipGramNegativeSampling` 有两套词向量表：`target_embeddings` 表示中心词，`context_embeddings` 表示上下文。`forward` 输入中心词 `[B]`、正上下文 `[B]`、负样本 `[B,K]`，输出可反向传播的标量损失；`word_vectors` 返回训练后的中心词表。运行 `python Word2Vec/word2vec.py`。例子直接给定词 ID 对，没有包含从语料构造窗口、频率降采样和负采样分布的预处理。

## 3. 数学解释

中心词向量为 `u`，真实上下文向量为 `v⁺`，负样本向量为 `v⁻ⱼ`。目标是最大化 `log σ(uᵀv⁺)+Σⱼ log σ(−uᵀv⁻ⱼ)`；源码取负号作为损失。真实配对的内积被推高，随机噪声配对被推低。`σ` 是 sigmoid。负采样避免对整个词表计算 softmax，单个样本只访问一个正词和 `K` 个负词。
