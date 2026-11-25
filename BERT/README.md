# BERT：双向文本编码器

## 1. 出处、提出者与意义

出处是 Jacob Devlin、Ming-Wei Chang、Kenton Lee 和 Kristina Toutanova 2018 年的 [《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](https://arxiv.org/abs/1810.04805)。作者用 Transformer 编码器同时读取词左右两边的上下文，先通过遮盖词预测（MLM）和下一句预测（NSP）预训练，再把表示用于问答、分类等任务。对于一句话里含义依赖前后文的词，双向读取尤其有用。

这让“先训练通用编码器、再适配具体任务”成为自然语言处理的核心做法之一。本项目实现双向编码器、词/位置/句段嵌入、MLM 与 NSP 两个输出头。它只定义模型；如何构造 `[MASK]`、正负句对、标签和大规模语料属于训练数据流程，不能把随机初始化的输出当作预训练结果。

## 2. 源码导航与直观解释

先看 [attention_core.py](attention_core.py) 的 `MultiHeadAttention`：Q/K/V 投影、拆头、按 PAD 掩码过滤和加权汇合都在项目内。再看 [bert.py](bert.py) 的 `BERTEncoder`：词 ID、位置 ID、句段 ID 的嵌入相加后进入多个 `BERTBlock`；`BERTForPretraining` 把每个位置的隐藏状态送到 MLM 词表头，并把第一个 `[CLS]` 位置的状态送到 NSP 二分类头。

输入 `token_ids` 为 `[B,T]`，`token_type_ids` 标记属于第一句还是第二句，`attention_mask` 标记有效词。输出 `mlm_logits` 为 `[B,T,V]`，`nsp_logits` 为 `[B,2]`。例如“这部电影很 [MASK]”中，预测被遮盖位置可同时参考左边的“电影”和右边的词；`BERTBlock` 只屏蔽 PAD，没有 GPT 的下三角未来掩码。仓库根目录运行 `python BERT/bert.py` 查看两个输出形状；[序列检查](../checks/test_sequence.py) 中修改后面的词会影响前面位置的表示，体现双向性。

## 3. 数学解释

输入位置 `t` 的初始向量是 `eₜ = E_token(xₜ)+E_position(t)+E_segment(sₜ)`：三个嵌入分别表示词是什么、在第几个位置、属于哪一句。`BERTEncoder.forward` 直接把它们相加，再做层归一化，得到 `[B,T,D]` 的表示。每个 `BERTBlock` 做多头自注意力与前馈变换，并在两步后用残差连接保留已有信息。

每个注意力头计算 `A=softmax(QKᵀ/√d + M)`、`H=AV`。`Q/K/V` 都来自同一输入句，`d` 是头宽度；`M` 只让 PAD 位置不能作为键被读取。有效位置之间没有方向限制，所以第一个位置可以从后面的词取信息，后面也可从前面取信息。这与 GPT 用下三角掩码预测下一个词的目标不同。源码的 `valid_key_mask` 最终传入 `MultiHeadAttention.forward`。

MLM 对**被选中预测的那些位置**计算 `L_MLM = −Σ_{t∈M} log p(xₜ|被遮盖的输入)`，`M` 是训练数据指定的预测位置集合；`mlm_decoder` 输出各位置 logits，但选哪些位置及交叉熵由调用方决定。NSP 把 `[CLS]` 的最终向量经 `pooler`、`nsp_classifier` 变为“相接/不相接”两个 logits，再做二分类交叉熵；原论文合并两项损失。代码把 MLM 输出权重与词嵌入权重绑定，使输入和预测共享同一词表参数。
