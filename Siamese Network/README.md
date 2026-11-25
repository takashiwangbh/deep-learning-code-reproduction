# Siamese Network：共享编码器比较两个样本

## 1. 出处、提出者与意义

Jane Bromley、Isabelle Guyon、Yann LeCun、Eduard Säckinger、Roopak Shah 在 1993 年的 [Signature Verification using a Siamese Time Delay Neural Network](https://papers.nips.cc/paper_files/paper/1993/hash/288cc0ff022877bd3df94bc9360b9c5d-Abstract.html) 中用两条共享权重的分支判断签名是否匹配。其重要思想是学习一个表征空间，使相似样本靠近、不相似样本远离；人脸验证、度量学习和对比学习都沿用了这一思路。

## 2. 源码导航与直观解释

[siamese.py](siamese.py) 只定义一套 `encoder`，`forward(left,right)` 对两张图重复调用同一对象，因此两边参数完全共享。输出是两组 `[B,E]` 嵌入；`contrastive_loss` 用成对标签 `same=1/0` 计算距离损失。运行 `python 'Siamese Network/siamese.py'` 可见损失与反向传播。为方便 CPU 教学，编码器是小 CNN，不是原论文的签名时间延迟网络。

## 3. 数学解释

两张输入得到 `z₁=f_θ(x₁)`、`z₂=f_θ(x₂)`，参数 `θ` 相同；距离 `d=‖z₁−z₂‖₂`。若 `y=1` 表示匹配、`y=0` 表示不匹配，示例损失为 `L=y d²+(1−y)max(0,m−d)²`。匹配对被拉近；不匹配对只在距离小于边界 `m` 时被推远。共享权重保证输入顺序不会改变单个样本的编码规则，且一次更新同时影响两条分支。

如果左右输入完全相同，共享编码器必然给出相同嵌入，此时匹配对的距离与损失都为零；这也是检查权重是否真正共享的最简单办法。若左右各自建一套参数，即使输入相同也未必得到同一向量。训练数据必须提供成对关系，否则这个损失无法知道哪些样本该靠近。
