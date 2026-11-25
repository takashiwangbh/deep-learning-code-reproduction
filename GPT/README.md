# GPT：从下一个词预测到逐词生成

## 1. 出处、提出者与意义

本项目对应 Alec Radford、Jeffrey Wu、Rewon Child、David Luan、Dario Amodei 和 Ilya Sutskever 2019 年的 [《Language Models are Unsupervised Multitask Learners》](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)，即 GPT-2 技术报告。它沿用 Transformer 的注意力模块，但只保留用于自回归生成的解码器：读到前面的词，预测下一个词。大量文本可以自动构造“前文→下一个词”的训练样本，无须人工逐句标注。

这条路线是后来生成式大语言模型的基础。本项目把网络缩小到可在 CPU 上阅读、运行的规模，保留学习式位置嵌入、因果注意力、残差块、逐词生成和 KV 缓存；没有论文的权重、数据规模或实际生成能力。`train_toy_gpt.py` 用内置重复短文展示完整训练步骤，输出仅作学习用途。

## 2. 源码导航与直观解释

先打开 [gpt.py](gpt.py)：`CausalSelfAttention.forward` 生成 Q/K/V，并用因果掩码阻止当前位置偷看未来；`GPTBlock` 把注意力与逐位置前馈网络串起来；`GPT.forward` 从词 ID 得到词表分数；`GPT.generate` 用这些分数贪心选择下一个词，并复用之前算好的 K/V。输入 `token_ids` 是 `[B,T]`，输出 logits 是 `[B,T,V]`，其中 `B` 是批量、`T` 是当前长度、`V` 是词表大小。logits 是每个词的未归一化分数。

再读 [train_toy_gpt.py](train_toy_gpt.py)：它从内置文本建立字符词表，把 `x` 的每个位置与右移一位的 `y` 对齐，用交叉熵训练，然后调用生成函数。好比输入“深 度 学”，标签是“度 学 习”，每个位置都在学“接下来是什么”。在仓库根目录运行 `python GPT/gpt.py` 可看形状与随机生成；运行 `python GPT/train_toy_gpt.py` 可看训练损失。对照 [序列检查](../checks/test_sequence.py) 可观察“修改未来词不改变过去输出”，以及一次性前向和分段 KV 缓存前向结果一致。

## 3. 数学解释

把序列写作 `x₁,…,xₜ`，自回归模型学习 `p(x₁,…,xₜ)=∏ₜ p(xₜ|x₁,…,xₜ₋₁)`。训练时 `train_toy_gpt.py` 的 `x` 与 `y` 相差一个位置，交叉熵最小化 `L=−Σₜ log p(yₜ|x₁,…,xₜ)`。这意味着一次前向能同时训练所有位置的“下一词”预测。`GPT.forward` 最后的 `lm_head` 把每个位置的隐藏向量映射到 `V` 个词的 logits；训练损失内部再把 logits 转成概率。

注意力的核心是 `softmax(QKᵀ/√d + M)V`。`Q` 表示当前位置想找什么，`K` 表示各位置能匹配什么，`V` 是要读取的内容，`d` 是每个注意力头的宽度；`M` 在未来位置给极小值，所以 softmax 后未来词的权重近似 0。`CausalSelfAttention.forward` 中 `allowed`、`masked_fill`、`softmax` 和 `weights @ v` 分别对应掩码、归一化及加权读取。多个头各自计算后再拼接，残差连接让输入信息能越过注意力与前馈层。

生成第 `t+1` 个词时，旧词的 K/V 不会改变。`past_key_values` 保存每层旧 K/V，新一步只为新词计算 Q/K/V，并把新 K/V 接到缓存后面；因此无需每次从头重算整个前缀。代码还用缓存长度确定新词的位置和因果掩码的行，使分段推理与整段推理对应。
