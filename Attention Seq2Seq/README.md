# 加性注意力 Seq2Seq

## 1. 出处、提出者与意义

出处是 Dzmitry Bahdanau、Kyunghyun Cho 与 Yoshua Bengio 2014 年的论文 [《Neural Machine Translation by Jointly Learning to Align and Translate》](https://arxiv.org/abs/1409.0473)，发表于 ICLR 2015。早期编码器—解码器先把整句输入压成**一个**固定向量，再用它生成译文；句子变长时，这个单一向量容易成为信息瓶颈。作者提出让解码器每产生一个词时，重新查看源句各位置，自动选出此刻相关的词。

它的重要性不只在机器翻译。后来图像、语音和文本模型大量使用的“注意力”都建立在同一个直觉上：**需要什么信息，就针对当前问题从一组候选信息中读取**。本文件保留加性打分和逐词解码的关键计算；为便于学习，编码器是单向 GRU，而不是论文中使用的双向编码器，不能直接对比论文成绩。

## 2. 源码导航与直观解释

先看 [gru_cell.py](gru_cell.py) 中的 `GRUCell`，它是本项目内的门控单元源码。然后打开 [attention_seq2seq.py](attention_seq2seq.py)：`encode` 把源句逐词变成一列状态，`AdditiveAttention.forward` 给这些状态打分并加权求和，`decode_step` 用当前词和读取到的上下文预测下一个词，`forward` 把这些步骤在整个目标句上展开；`greedy_decode` 展示逐词生成。

输入 `source` 为 `[B,S]` 的源词 ID，`target_input` 为 `[B,T]` 的目标词 ID（开头通常是 BOS），输出是 `[B,T,V目标]` 的未归一化词表分数。`encode` 还返回有效位置掩码；注意力不会把填充用的 PAD 当成真实词。想象翻译“我 喜欢 苹果”：生成“apple”时，模型可以把更大的注意力权重放在“苹果”上。仓库根目录运行 `python 'Attention Seq2Seq/attention_seq2seq.py'` 即可检查输出形状；模型尚未训练，生成内容没有翻译能力。

## 3. 数学解释

令 `hₛ` 为源句第 `s` 个词的编码状态，`dₜ₋₁` 为生成第 `t` 个词之前的解码状态。`AdditiveAttention.forward` 首先计算相关性 `eₜₛ = vᵀ tanh(Wq dₜ₋₁ + Wk hₛ)`：两个线性层分别变换“当前要问什么”和“源位置能提供什么”，相加后由 `score` 变成一个数字。`e` 越大，该源词越可能有用。

随后用 `aₜₛ = softmaxₛ(eₜₛ)` 把所有源位置分数转成总和为 1 的权重；对 PAD 位置先遮蔽，使它权重为 0。上下文向量为 `cₜ = Σₛ aₜₛ hₛ`，在代码里对应 `torch.bmm`。这不是随机选择一个源词，而是把所有源状态按权重混合，因而整个流程可以反向传播。

最后 `decode_step` 把当前目标词向量与 `cₜ` 拼在一起，送入 GRU 得到 `dₜ`，再由线性层输出下一词 logits。训练时将目标句右移：输入是 `<BOS>, y₁, …`，标签是 `y₁, y₂, …`。阅读 `greedy_decode` 时可看到同一公式在推理中不断重复，只是下一步词由当前 logits 的 `argmax` 决定。
