# ELMo：同一个词在不同上下文有不同向量

## 1. 出处、提出者与意义

Matthew E. Peters、Mark Neumann、Mohit Iyyer 等人在 2018 年提出 [《Deep contextualized word representations》](https://arxiv.org/abs/1802.05365)。Word2Vec 或 GloVe 给每个词一条固定向量，难以同时表示一个词在不同句子中的不同含义。ELMo 先训练正向与反向语言模型，再把它们各层的隐藏状态组合成下游任务使用的词向量。这样，一个位置的表示既看左边也看右边，而且任务可以学习浅层和深层信息各占多少。

## 2. 源码导航与直观解释

[elmo.py](elmo.py) 的 `ELMo` 包含互不共享的 `forward_layers`、`backward_layers`：前者从左到右读句子，后者从右到左读。`contextual_embeddings` 收集词嵌入以及两方向每层 LSTM 输出，用 `layer_logits.softmax()` 学习各层权重，再乘可训练的 `gamma`；输入 token `[B,T]`，输出上下文向量 `[B,T,2H]`。`language_model_loss` 用正向状态预测下一个词，用反向状态预测上一个词；为了防止目标泄漏，两路训练输入分别去掉不能看的那一端。根目录运行 `python ELMo/elmo.py`。本例用词 ID 嵌入替代论文的字符 CNN，并缩小 LSTM，未含大语料预训练。

## 3. 数学解释

对词位置 `t`，正向语言模型估计 `p(w_t|w_1,…,w_{t-1})`，反向模型估计 `p(w_t|w_{t+1},…,w_T)`；两个交叉熵相加训练各自的 LSTM。`language_model_loss` 的移位输入与目标正对应这两个条件概率。下游表示把第 `j` 层双向状态 `h_{t,j}=[\overrightarrow{h}_{t,j};\overleftarrow{h}_{t,j}]` 混合：`ELMo_t=γ Σ_j softmax(s)_j h_{t,j}`。源码中的 `representations`、`layer_logits`、`gamma` 与公式逐项对应。`s` 属于任务层参数，所以同一预训练双向 LM 用于不同任务时，可以偏好不同深度的信息。
