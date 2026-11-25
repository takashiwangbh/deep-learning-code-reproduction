# RoBERTa：更充分训练的 BERT 式编码器

## 1. 出处、提出者与意义

Yinhan Liu、Myle Ott、Naman Goyal 等人在 2019 年提出 [《RoBERTa: A Robustly Optimized BERT Pretraining Approach》](https://arxiv.org/abs/1907.11692)。这项工作重新检验 BERT 的训练设置，指出训练时间、数据规模、批量大小和遮盖方式能显著影响结果。RoBERTa 去掉下一句预测任务，使用动态遮盖：同一段文字在不同训练轮次可能被遮住不同位置。它说明评估新模型时，必须先给强基线合理的训练预算和数据处理。

## 2. 源码导航与直观解释

[roberta.py](roberta.py) 中 `RoBERTa.forward` 是双向 Transformer 编码器，输入 token ID `[B,T]`，输出每个位置的词表 logits `[B,T,V]`。`dynamic_mask` 每次调用重新抽取约 15% 的非特殊 token：所选位置中约 80% 换成 MASK、10% 换随机词、10% 保持原词；标签只在所选位置保留，其他位置为 `-100`。`training_loss` 每批重新遮盖，再只对选中的词计算交叉熵。运行 `python RoBERTa/roberta.py`。小序列会强制至少选中一处，以免全是忽略标签导致 NaN；这个教学处理不等于论文的大批量掩码分布。本例没有大语料、BPE 分词或训练日程，也没有 NSP 头。

## 3. 数学解释

设被选位置集合为 `M`，双向编码器得到每个位置的分布 `p_θ(w|corrupt(x))`，目标 `L_MLM=-(1/|M|)Σ_{t∈M} log p_θ(x_t|corrupt(x))`。`dynamic_mask` 返回的 `labels=-100` 使源码只在 `M` 求损失；`mask_token`、随机替换和原词保留分别是训练输入的三种扰动。动态遮盖意味着每次进入 `training_loss` 都重新采样 `M`，比把遮盖位置提前写死更充分地利用同一文本。这里的核心研究结论不是更复杂的注意力公式，而是强训练配方；源码专注展示可独立运行的模型结构与动态 MLM 通路。
