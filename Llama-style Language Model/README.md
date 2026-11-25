# Llama 风格语言模型：RoPE、GQA、RMSNorm 与 SwiGLU

## 1. 出处、提出者与意义

本项目参照 Hugo Touvron、Louis Martin、Kevin Stone 等人 2023 年的 [《Llama 2: Open Foundation and Fine-Tuned Chat Models》](https://arxiv.org/abs/2307.09288) 所代表的现代解码器路线。关键组件各有出处：Jianlin Su、Yu Lu、Shengfeng Pan、Bo Wen、Yunfeng Liu 等人 2021 年的 [RoFormer](https://arxiv.org/abs/2104.09864) 提出旋转位置编码；Joshua Ainslie、James Lee-Thorp、Michiel de Jong、Yury Zemlyanskiy、Federico Lebrón、Sumit Sanghai 等人 2023 年的 [GQA](https://arxiv.org/abs/2305.13245) 提出分组查询注意力；Biao Zhang 与 Rico Sennrich 2019 年提出 [RMSNorm](https://arxiv.org/abs/1910.07467)；Noam Shazeer 2020 年的 [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) 研究了包括 SwiGLU 在内的门控前馈变体。

这些设计分别解决位置表达、生成时 KV 缓存体积和深层网络计算稳定性等问题：RoPE 把位置信息写进 Q/K 的旋转，GQA 让多个查询头共享较少的 K/V 头，RMSNorm 与 SwiGLU 构成常见的现代块。本文件是展示这些结构如何组合的**教学模型**，并非任何已发布 Llama 模型的精确结构、权重或性能复现，也不包含分词器和训练数据。

## 2. 源码导航与直观解释

全部源码在 [llama_style.py](llama_style.py)。按 `RMSNorm` → `apply_rope` → `GroupedQueryAttention` → `SwiGLU` → `LlamaBlock` → `LlamaStyleLM.forward` 阅读。`apply_rope` 随 token 位置旋转 Q/K 的成对通道；`GroupedQueryAttention` 建立较少 K/V 头，然后在实际注意力计算时重复它们；`LlamaBlock` 用注意力与门控前馈更新隐藏状态。输入词 ID 为 `[B,T]`，输出 logits 为 `[B,T,V]`。`use_cache=True` 时额外返回每层 K/V，形状中的头数为 `num_kv_heads`，小于查询头数 `num_heads`。

直观地看，RoPE 给每个词的“查询”和“索引”加上与位置相关的方向；GQA 像几位读者共用一套索引卡片，少存卡片仍可分别提问；SwiGLU 让一条分支控制另一条分支传递多少信息。仓库根目录运行 `python 'Llama-style Language Model/llama_style.py'` 查看输出形状。[序列检查](../checks/test_sequence.py) 验证未来词不会影响过去，以及分段 KV 缓存结果与整段计算相符。

## 3. 数学解释

`RMSNorm` 对向量 `x∈Rᴰ` 计算 `RMSNorm(x)=γ ⊙ x / √((1/D)Σᵢxᵢ²+ε)`：`D` 是隐藏宽度，`γ` 是可学习的逐维缩放，`ε` 防止除零。它按均方根缩放，不减去均值；在 `RMSNorm.forward` 中可看到平方、按最后一维求均值、倒平方根和乘权重。

RoPE 把每两个通道看成二维坐标。对位置 `t` 和通道对 `i`，角度为 `θₜᵢ=t·10000^(−2i/d)`，并用旋转矩阵 `[[cosθ,−sinθ],[sinθ,cosθ]]` 作用于 Q/K 的这两个数；`d` 是注意力头宽度。`apply_rope` 的 `even/odd`、`cosine/sine` 正是这两条旋转公式。位置不同，Q 与 K 的匹配分数就随相对位置变化。

GQA 有 `Hq` 个查询头和 `Hkv` 个 K/V 头，且 `Hq` 能被 `Hkv` 整除。每组 `Hq/Hkv` 个 Q 头共用一对 K/V 头；`repeat_interleave` 只在计算注意力时展开 K/V，缓存仍保存 `[B,Hkv,T,d]`，而普通多头缓存需 `[B,Hq,T,d]`。每头依然计算 `softmax(QKᵀ/√d + 因果掩码)V`，所以当前位置不会读未来。`SwiGLU` 则计算 `W_down(SiLU(W_gate x) ⊙ W_up x)`：`SiLU(z)=z·sigmoid(z)`，乘号 `⊙` 是逐元素相乘；在源码的 `SwiGLU.forward` 中两条上投影分支与下投影逐一对应。
