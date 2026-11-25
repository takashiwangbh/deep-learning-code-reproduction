# T5：把语言任务统一写成文本到文本

## 1. 出处、提出者与意义

Colin Raffel、Noam Shazeer、Adam Roberts 等人在 2019 年的 [《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》](https://arxiv.org/abs/1910.10683) 提出 T5。它把翻译、问答、摘要甚至分类都写成“输入一段文字，输出一段文字”：例如输入 `translate English to German: ...`，输出德语句子。这样同一编码器—解码器结构和训练接口就能服务不同任务。论文还研究了预训练目标、数据和规模对迁移的影响；核心的 span corruption 是把连续词片段替换为哨兵标记，再让解码器还原缺失片段。

## 2. 源码导航与直观解释

打开 [t5.py](t5.py)：`relative_bucket` 把词间距离分桶，近距离分得细、远距离分得粗；`RMSNorm` 只按均方根调整特征大小；`Attention` 明写 Q/K/V、缩放点积和掩码；`Block` 先做注意力再做逐词前馈；`T5` 用同一词嵌入接编码器、解码器，并与输出层共享权重。编码器读完整源句，解码器只看已给的目标前缀，同时可交叉读取源句。输入是 `source_ids [B,S]` 和 `decoder_ids [B,T]`，返回 `[B,T,V]` 的下一词分数。从根目录运行 `python T5/t5.py`。

这是展示结构的 CPU 小模型：没有实现论文的大规模 C4 预训练、span corruption 数据管线、完整解码策略和分布式训练。解码输入须由训练者右移目标序列；源码底部仅用小样本演示损失和反向传播。

## 3. 数学解释

每个注意力头先算 `Q=XW_Q, K=XW_K, V=XW_V`，再用 `softmax(QKᵀ/√d_k+B)V` 汇总信息。这里 `B` 不是绝对位置向量，而是由 query 与 key 的相对距离查表得到的偏置；`relative_bucket` 保留近距离差异，同时把很远的位置归入较少的桶。解码器的因果掩码让第 `t` 个位置只能读 `≤t` 的目标词；交叉注意力让它能读所有有效源词。与普通 LayerNorm 不同，`RMSNorm(x)=x/√(mean(x²)+ε)·γ` 不先减均值。训练时常用逐 token 交叉熵 `−log p(y_t | y_<t, source)`，源码展示了 logits 的生成，真实训练应对 PAD 标签屏蔽并做目标右移。
