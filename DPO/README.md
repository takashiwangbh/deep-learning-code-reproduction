# Direct Preference Optimization（DPO）

## 1. 出处、提出者与意义

Rafael Rafailov、Archit Sharma、Eric Mitchell、Stefano Ermon、Christopher D. Manning 和 Chelsea Finn 在 [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)（2023）中提出 DPO。训练数据不是单一“正确答案”，而是同一个问题下哪条回答更受偏好的成对比较。DPO 把这类偏好直接转成语言模型的分类式损失，让模型提高获选回答相对于被拒绝回答的概率。

本项目只实现目标函数和逐 token 概率汇总，不包含完整的数据收集、语言模型或大规模微调流程。先理解 GPT 项目的 token 概率，再读这里的偏好损失会更顺畅。

## 2. 源码导航与直观解释

[dpo.py](dpo.py) 的 `sequence_log_probability` 从 `[B,T,V]` 的模型分数中，取出目标 token 的对数概率并沿回答长度求和；`mask` 用来只统计回答，不统计提示词和填充位置。它支持把被忽略的目标 ID 写为 `-100`。`dpo_loss` 接收策略模型对“获选/被拒绝”两条回答的 log 概率，以及冻结参考模型对应的两个值，输出一个标量损失。

从根目录运行 `python 'DPO/dpo.py'`，看一次示例的损失和梯度。真正训练时，先用模型分别得到两条回答的逐 token logits，并正确做下一个 token 的位置对齐，再调用 `sequence_log_probability`；参考模型的输出不参与梯度更新。源码刻意只保留核心算法，让人能看出偏好比较究竟落在哪四个数字上。

## 3. 数学解释

记 `πθ(y⁺|x)` 为当前模型在问题 `x` 下生成获选回答 `y⁺` 的概率，`y⁻` 为被拒绝回答；`πref` 是固定参考模型。先定义策略模型的对数概率差 `Δθ = logπθ(y⁺|x)−logπθ(y⁻|x)`，参考模型同理得 `Δref`。DPO 损失为 `L = −log σ(β(Δθ−Δref))`，其中 `σ` 是 sigmoid，`β>0` 控制相对于参考模型改变偏好的强度。`dpo_loss` 里的 `log_ratio` 就是括号中的 `Δθ−Δref`。

一条回答由多个 token 组成，其整体 log 概率是各 token 对数概率之和：`logπ(y|x)=Σₜ logπ(yₜ|x,y_{<t})`。`sequence_log_probability` 中 `log_softmax → gather → mask → sum` 正对应这一步。使用对数是因为一串小概率相乘容易数值下溢，而对数把乘法变成加法。若当前模型比参考模型更偏向 `y⁺`，括号变大，损失变小。
