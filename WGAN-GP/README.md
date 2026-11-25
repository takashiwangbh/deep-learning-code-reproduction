# WGAN-GP：用梯度惩罚稳定对抗训练

## 1. 出处、提出者与意义

Ishaan Gulrajani、Faruk Ahmed、Martin Arjovsky、Vincent Dumoulin 与 Aaron Courville 在 2017 年 [《Improved Training of Wasserstein GANs》](https://arxiv.org/abs/1704.00028) 中指出，早期 WGAN 用权重裁剪约束判别器会损伤表达能力。他们改为在真实与生成样本之间的插值点惩罚梯度范数，让 critic 更接近 1-Lipschitz 函数。这是理解 Wasserstein 对抗目标与现代 GAN 训练稳定性的重要方法。

## 2. 源码导航与直观解释

[wgan_gp.py](wgan_gp.py) 的 `Generator` 从 `[B,8]` 噪声产生 `[B,16]` 向量；`Critic` 返回实数分数而非概率，因此末层**没有 sigmoid**。`gradient_penalty` 为每个样本随机取插值系数 `α`，得到真实点与假点之间的位置，再对 critic 输出相对于该位置求梯度。`critic_loss` 结合真假得分差和惩罚，`generator_loss` 让假样本的 critic 得分升高。运行 `python WGAN-GP/wgan_gp.py`。这里为看清公式使用 MLP 和随机向量；论文可用卷积网络训练图像。

## 3. 数学解释

在 Wasserstein 对偶形式里，critic 要在 1-Lipschitz 约束下最大化 `E[D(x_real)]−E[D(x_fake)]`。因为优化器通常最小化目标，源码写作 `E[D(fake)]−E[D(real)] + λ E[(‖∇_{x̂}D(x̂)‖₂−1)²]`，其中 `x̂=αx_real+(1−α)x_fake` 且 `α∼U(0,1)`。`torch.autograd.grad(..., create_graph=True)` 必不可少：惩罚的外层反向传播还要穿过“对输入求梯度”这一步。生成器最小化 `−E[D(G(z))]`。代码故意不做权重裁剪；梯度惩罚正是它的替代。
