# ULMFiT：语言模型预训练再逐层微调

## 1. 出处、提出者与意义

Jeremy Howard 和 Sebastian Ruder 在 2018 年提出 [《Universal Language Model Fine-tuning for Text Classification》](https://arxiv.org/abs/1801.06146)。在 Transformer 大规模预训练成为常态之前，ULMFiT 已证明通用语言模型可以先读无标签文本，再适应目标领域，最后用于文本分类。论文特别关注“微调时别把预训练知识一下子冲掉”：采用逐层解冻、不同层不同学习率、先快升后慢降的学习率日程等方法。它是 NLP 迁移学习的重要基线。

## 2. 源码导航与直观解释

[ulmfit.py](ulmfit.py) 的 `language_model_loss` 用词前缀预测下一词，代表第一阶段通用预训练和第二阶段领域适应。`classify` 把最终 LSTM 输出的最后状态、时间最大值、时间平均值拼接成分类特征，代表第三阶段。`gradual_unfreeze(stage)` 从只训练分类头开始，先解冻最高一层 LSTM，再逐层往下；`discriminative_groups` 给接近输出的层更大学习率；`slanted_triangular_lr` 返回快速上升、缓慢下降的比例。输入 `[B,T]`，分类 logits `[B,C]`；运行 `python ULMFiT/ulmfit.py`。示例采用普通小型 LSTM，未实现论文 AWD-LSTM 的所有 dropout 细节、BPTT 日程或真实语料训练。

## 3. 数学解释

语言模型最小化 `L_LM=-Σ_t log p(w_t|w_{<t})`；分类阶段最小化标签的交叉熵。逐层解冻可写为阶段集合 `S_0={分类头}`、`S_1=S_0∪{顶层LSTM}`，依次把更低层加入可训练集合；源码设置 `.requires_grad_`。判别学习率让第 `l` 层使用 `η_l=η_top/decay^(距顶层层数)`，越靠近词嵌入，步子越小。倾斜三角日程在训练的前一小段把学习率从 `η/ratio` 升到 `η`，余下训练逐步降回 `η/ratio`；`slanted_triangular_lr` 返回该倍数。三种机制共同让新任务学得足够快，同时减少灾难性遗忘。
