# PixelCNN：按顺序预测每个像素

## 1. 出处、提出者与意义

Aaron van den Oord、Nal Kalchbrenner 与 Koray Kavukcuoglu 在 2016 年 [《Pixel Recurrent Neural Networks》](https://arxiv.org/abs/1601.06759) 中同时研究 PixelRNN 和更适合并行计算的 PixelCNN。PixelCNN 把图像概率拆成每个像素在此前像素给定时的条件概率。训练时整张图可并行求损失；采样时必须按从上到下、从左到右的顺序逐像素生成。它说明“卷积”也能做严格自回归建模。

## 2. 源码导航与直观解释

打开 [pixelcnn.py](pixelcnn.py) 的 `MaskedConv2d`：掩码 A 用在输入层，挡住当前位置及未来位置；掩码 B 用在后续隐藏层，允许当前位置的**隐藏特征**参与，但它仍只包含过去的原始像素。`PixelCNN.forward` 输出 `[B,1,H,W]` 二值像素 logits，`loss` 用 Bernoulli BCE，`sample` 循环生成每一格。根目录运行 `python PixelCNN/pixelcnn.py`。本例用二值灰度图帮助看清掩码；论文主要评估多通道、多取值的真实图像，需要更复杂的输出分布。

## 3. 数学解释

把二维图像按行展开为 `x₁,…,xₙ`，联合概率可精确分解为 `p(x)=∏ᵢ p(xᵢ|x₁,…,xᵢ₋₁)`，负对数似然是这些条件概率的 `−log` 之和。掩码卷积让位置 `i` 的 logits 不接触 `j≥i` 的原始像素，从结构上保证条件分解成立；`MaskedConv2d` 将对应权重直接乘 0，而非只在损失中忽略它们。二值图的每个条件分布是 `Bernoulli(sigmoid(logit_i))`，因此 `binary_cross_entropy_with_logits` 就是逐像素负对数似然。修改一个未来像素不应改变较早位置 logits；检查脚本会验证这个因果性质。
