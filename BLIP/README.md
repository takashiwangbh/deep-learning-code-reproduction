# BLIP：图文理解与图像描述共用的预训练框架

## 1. 出处、提出者与意义

Junnan Li、Dongxu Li、Caiming Xiong、Steven Hoi 在 2022 年提出 [《BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation》](https://arxiv.org/abs/2201.12086)。一些图文模型擅长检索却不善于写描述，另一些能生成文本却难以判断图文是否相符。BLIP 将图文对比、图文匹配和图像条件语言建模放在同一框架，并提出 CapFilt：先给网络图片生成描述，再过滤噪声较大的图文配对。它是理解与生成结合的一项代表工作。

## 2. 源码导航与直观解释

[blip.py](blip.py) 的 `VisionEncoder` 把图片切成视觉 token，`encode_text` 编码文本 token。`BLIP.forward` 同时计算三个训练任务：`itc` 在 batch 内把正确图文配对拉近，`itm` 用文本对图像的交叉注意力判断匹配，`lm` 用带因果掩码的文本解码器从图像生成后续词。`generate` 逐词生成描述，`bootstrap_and_filter` 再用匹配概率筛选生成的描述，展示 CapFilt 的两步。输入图片 `[B,3,32,32]` 和带 BOS/EOS 的 token `[B,T]`，批次至少两对；从根目录运行 `python BLIP/blip.py`。随机参数生成的词及筛选分数没有语义用途；真实 CapFilt 需要已训练的 captioner/filter、更大数据和噪声处理。

## 3. 数学解释

ITC 先归一化图像向量 `u_i` 与文本向量 `v_j`，分数 `s_ij=u_i·v_j/τ`，再用第 `i` 张图对应第 `i` 条文本的双向交叉熵训练。ITM 给真实 `(image_i,text_i)` 标 1、批内错配标 0；文本查询 `Q` 通过 `softmax(QKᵀ/√d)V` 从图像 token 的 `K,V` 读取视觉证据，再由 `matching_head` 分类。LM 采用教师强制：给定图像和前 `t` 个 token，预测下一个 token，损失 `-Σ_t log p(w_t|w_{<t},image)`；源码的上三角 `causal` 掩码阻止偷看未来词。CapFilt 先用 LM 采样描述，再根据 ITM 的匹配概率是否超过阈值保留图文对；本例以贪心生成简化采样。
