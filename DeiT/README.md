# DeiT：让视觉 Transformer 从教师模型学习

## 1. 出处、提出者与意义

Hugo Touvron、Matthieu Cord、Matthijs Douze 等人在 2020 年提出 [《Training data-efficient image transformers & distillation through attention》](https://arxiv.org/abs/2012.12877)。早期 ViT 通常依赖很大的预训练数据；DeiT 展示了更有效的训练策略，并引入可通过注意力与图像块交互的**蒸馏 token**。学生网络既用普通分类 token 学真实标签，也用蒸馏 token 学教师判断。这样教师知识进入整个 Transformer，而不仅是训练结束时匹配输出。

## 2. 源码导航与直观解释

[deit.py](deit.py) 中 `Attention` 显式计算多头 Q/K/V；`Block` 由注意力、前馈与两条残差组成；`DeiT` 把图像切成 patch token，再在开头放两个可学习 token：`cls` 和 `distill`。网络输出两个 `[B,C]` 分类头，推理可平均两头分数；`hard_distillation_loss` 用真实类别监督前者，用教师 logits 的最大类监督后者。输入 `[B,3,H,W]`，默认 32×32 教学图，根目录运行 `python DeiT/deit.py` 可看两个头与反向传播。

源码忠实展示硬蒸馏 token 路线；教师网络在外部提供 logits，没有在此复制或训练教师，也没有论文的数据增强、优化日程和大规模预训练。若要训练，需给函数真实标签与一个固定教师的输出。

## 3. 数学解释

每张图切出 `N=(H/P)(W/P)` 个块，映射成宽度 `D` 的 token，再拼接 `z_cls,z_dist`。多头注意力允许两个特殊 token 从全部 patch 读取信息，最终各自经过线性头得到 `p_cls,p_dist`。硬蒸馏目标为 `L=(1−α)CE(p_cls,y)+αCE(p_dist,argmax teacher(x))`；源码中的 `teacher_logits.detach()` 保证学生更新不反传进教师。推理时两头 logits 可取均值。两头的参数分开，使学生可同时保留真实标注和教师给出的类别信息。
