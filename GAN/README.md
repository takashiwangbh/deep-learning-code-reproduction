# GAN：生成器与判别器的对抗游戏

## 1. 出处、提出者与意义

Ian Goodfellow、Jean Pouget-Abadie、Mehdi Mirza、Bing Xu、David Warde-Farley、Sherjil Ozair、Aaron Courville 与 Yoshua Bengio 在 2014 年 [《Generative Adversarial Networks》](https://arxiv.org/abs/1406.2661) 中提出 GAN。生成器试图把随机噪声变成像真实数据的样本，判别器试图区分真假；两者交替提升。它为后来的条件生成、图像翻译和高保真图像合成建立了共同框架。这个目录实现论文最基础的全连接版本，而不是把 DCGAN 当作原始 GAN。

## 2. 源码导航与直观解释

在 [gan.py](gan.py) 中，`Generator` 接受 `[B,8]` 噪声，输出 `[B,16]` 假样本；`Discriminator` 把 `[B,16]` 样本映射到 `[B,1]` 真伪 logits。读 `discriminator_loss` 时注意 `fake.detach()`：训练判别器这一轮不改变生成器。下一轮用 `generator_loss` 让假样本获得更高的真样本分数；实际训练应给 G、D 分别设置优化器并交替 `step()`。根目录运行 `python GAN/gan.py` 只演示前反向计算，不会从随机权重产生有意义的样本。

## 3. 数学解释

原始目标是 `min_G max_D E_{x∼p_data} log D(x) + E_{z∼p_z} log(1−D(G(z)))`。判别器希望真实样本得分高、假样本得分低。代码内部输出 logits `a`，用 `softplus(−a)=−log sigmoid(a)` 与 `softplus(a)=−log(1−sigmoid(a))` 稳定计算二分类损失。生成器采用常用的**非饱和**目标 `−E log D(G(z))`，即 `softplus(−D_logit(G(z)))`；它与原始 minimax 在理想平衡点一致，但生成器很差时梯度更有用。两条网络都显式写在本目录，模型训练不是调用现成 GAN API。
