# 序列模型与大语言模型：从循环网络到 RoPE

这里的模型**由本仓库用 PyTorch 基础张量与层重新实现**，没有用 `nn.LSTM`、`nn.Transformer`、`nn.MultiheadAttention` 或 `transformers` 包代替论文中的核心结构。每个 `.py` 文件的 `__main__` 都能用随机小输入执行；随机初始化的输出仅用于检查计算图。

## 学习顺序

| 顺序 | 文件 | 先理解什么 | 关键公式或代码入口 |
| --- | --- | --- | --- |
| 1 | [`recurrent.py`](recurrent.py) | 时间展开、隐藏状态、RNN/LSTM/GRU 的门 | `LSTMCell.forward` 中 `c_t=f_t*c_prev+i_t*g_t`；`GRUCell.forward` 中更新门混合新旧状态 |
| 2 | [`attention_seq2seq.py`](attention_seq2seq.py) | 编码器—解码器与加性注意力 | `AdditiveAttention.forward`：对每个源位置计算相关性，再加权求和 |
| 3 | [`transformer.py`](transformer.py) | 缩放点积、多头、位置编码、交叉注意力 | `MultiHeadAttention.forward`：`softmax(QKᵀ/√d_k)V` |
| 4 | [`gpt.py`](gpt.py) | 只用解码器的自回归模型、因果掩码、KV cache | `CausalSelfAttention.forward`；`GPT.generate` |
| 5 | [`bert.py`](bert.py) | 双向编码器、MLM 与原论文中的 NSP | `BERTEncoder.forward` 无下三角掩码；`BERTForPretraining.forward` |
| 6 | [`llama_style.py`](llama_style.py) | RoPE、GQA、RMSNorm、SwiGLU 等现代语言模型结构 | `apply_rope`；`GroupedQueryAttention.forward` |
| 7 | [`lora.py`](lora.py) | 低秩参数高效微调 | `LoRALinear.forward`：`Wx + (α/r)BAx` |

**张量形状约定**：`B` 是批量，`T` 是序列长度，`S` 是源序列长度，`D` 是隐藏维度，`H` 是注意力头数，`V` 是词表大小。所有模型的输入 token ID 为整数张量 `[B,T]`，输出通常是未经过 softmax 的 `[B,T,V]` logits。训练下一个 token 时，输入取序列 `[:-1]`，标签取 `[1:]`。

## 直接运行

在有 PyTorch 的 Python 环境中：

```bash
cd Deep-Learning-Sourcebook/sequence
python recurrent.py
python attention_seq2seq.py
python transformer.py
python gpt.py
python bert.py
python llama_style.py
python lora.py
python -m unittest -v test_sequence.py
python train_toy_gpt.py
```

`train_toy_gpt.py` 是完整但极小的字符级训练例子，内置重复文本，不下载数据：它展示字符到 ID、右移标签、交叉熵、反向传播、优化器更新和生成。其结果没有实际语言能力。实际研究还需要 tokenizer、真实语料、数据清洗、训练配方与评估。

## 论文与实现对应

- **循环网络**：Elman, *Finding Structure in Time* (1990), [论文](https://doi.org/10.1207/s15516709cog1402_1)；Hochreiter & Schmidhuber, *Long Short-Term Memory* (1997), [论文](https://doi.org/10.1162/neco.1997.9.8.1735)；Cho et al., *Learning Phrase Representations using RNN Encoder–Decoder* (2014), [论文](https://arxiv.org/abs/1406.1078)。`recurrent.py` 手动写出三个 cell，而非调用封装好的循环层。
- **加性注意力**：Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate* (2014), [论文](https://arxiv.org/abs/1409.0473)。`attention_seq2seq.py` 实现源序列对齐权重，源 PAD 位置被遮盖。
- **原始 Transformer**：Vaswani et al., *Attention Is All You Need* (2017), [论文](https://arxiv.org/abs/1706.03762)。`transformer.py` 包含 encoder 自注意力、decoder 因果自注意力和对 encoder 的交叉注意力。
- **GPT 式语言模型**：Radford et al., *Language Models are Unsupervised Multitask Learners* (2019), [论文](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)。`gpt.py` 展示 decoder-only、学习式位置嵌入和推理 KV cache；用较小宽度方便 CPU 阅读和运行。
- **BERT**：Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018), [论文](https://arxiv.org/abs/1810.04805)。`bert.py` 包含 MLM 和原论文的 NSP 输出；语料构造与 mask 策略需在训练端完成。
- **现代 LLM 结构**：Su et al., *RoFormer* (RoPE, 2021), [论文](https://arxiv.org/abs/2104.09864)；Ainslie et al., *GQA* (2023), [论文](https://arxiv.org/abs/2305.13245)；Touvron et al., *Llama 2* (2023), [论文](https://arxiv.org/abs/2307.09288)。`llama_style.py` 是展示这些结构如何组合的教学版，不声称重现任何完整发布模型的权重、训练或性能。
- **LoRA**：Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021), [论文](https://arxiv.org/abs/2106.09685)。`lora.py` 冻结基础线性层，只训练 A/B，并展示推理时合并权重。

## 读源码时可做的实验

1. 在 `recurrent.py` 的 `GRUCell.forward` 中将更新门设为常数 0 或 1，观察隐藏状态如何改变。
2. 在 `transformer.py` 去掉解码器的下三角掩码，再运行测试，观察未来 token 如何泄漏到过去位置。
3. 对比 `gpt.py` 的整段前向与分段 KV cache 前向，`test_sequence.py` 已验证两者一致。
4. 将 `llama_style.py` 的 `num_kv_heads` 从 2 改为 4，比较缓存张量的头数。
5. 在 `lora.py` 初始化后检查 B 全零，再给 B 一个更新，比较合并前后的输出。
