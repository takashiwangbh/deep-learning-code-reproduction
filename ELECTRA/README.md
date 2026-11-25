# ELECTRA：判断每个词是否被替换

## 1. 出处、提出者与意义

Kevin Clark、Minh-Thang Luong、Quoc V. Le、Christopher D. Manning 在 2020 年提出 [《ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators》](https://arxiv.org/abs/2003.10555)。BERT 的 MLM 只对少数被遮盖位置算预测损失；ELECTRA 先让一个小生成器给遮盖位置猜词，再让判别器逐位置判断输入词是否真的被替换。这样几乎每个非 PAD token 都为判别器提供训练信号，提高预训练样本效率。

## 2. 源码导航与直观解释

[electra.py](electra.py) 有两套独立双向 `TextEncoder`：小 `generator` 和较宽的 `discriminator`。`training_loss` 每批随机挑选部分非特殊词，先换成 MASK 交给生成器；生成器只在选中的位置预测原词。随后从生成器分布采样词填回，判别器读取**无 MASK 的替换句子**，给每个位置输出真假 logit。若生成器刚好猜中原词，该位置标签应为“未替换”；源码通过 `corrupted.ne(clean)` 计算实际标签，而不是直接把“被选中”当假词。运行 `python ELECTRA/electra.py`。这里未实现论文的嵌入共享、分布式大语料训练与下游微调，仅展示生成—替换—检测的核心。

## 3. 数学解释

令被选位置为 `M`，生成器损失 `L_G=-Σ_{t∈M} log p_G(x_t|MASKed(x))`。从生成器分布采样 `\tilde{x}_t` 后，判别标签 `y_t=1[\tilde{x}_t≠x_t]`；未选位置自然为 0。判别器损失对所有有效词计算 `L_D=-Σ_t[y_t log σ(D(\tilde{x})_t)+(1-y_t)log(1-σ(D(\tilde{x})_t))]`，源码用稳定的 `binary_cross_entropy_with_logits` 实现。总损失 `L=L_G+λL_D`，示例用 `λ=50` 突出判别学习。采样词是离散操作，判别器梯度不会穿过采样步骤回到生成器；生成器由自己的 MLM 损失训练。
