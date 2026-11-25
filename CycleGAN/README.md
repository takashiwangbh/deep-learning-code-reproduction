# CycleGAN：没有成对图片也能学习双向翻译

## 1. 出处、提出者与意义

Jun-Yan Zhu、Taesung Park、Phillip Isola 与 Alexei A. Efros 在 2017 年 [《Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks》](https://arxiv.org/abs/1703.10593) 中研究无配对图像翻译。若只有一批 A 域图片和一批 B 域图片，普通 pix2pix 无法知道哪张 A 对应哪张 B。CycleGAN 学两条相反路线，并要求“过去再回来”接近原图，借此减少只满足目标域外观却丢失内容的解。

## 2. 源码导航与直观解释

[cyclegan.py](cyclegan.py) 中 `ResnetGenerator` 负责 A→B 或 B→A；`PatchDiscriminator` 分别判断 A 与 B 域的真假。`losses` 生成 `fake_b=g_ab(real_a)`、`fake_a=g_ba(real_b)`，随后计算 `cycle_a=g_ba(fake_b)` 和 `cycle_b=g_ab(fake_a)`。它还包含恒等映射损失：目标本来就在 B 域时，`g_ab(real_b)` 不应乱改。输入、输出均为 `[B,3,16,16]`；运行 `python CycleGAN/cyclegan.py`。本例用小网络直观呈现训练目标，未包含论文的图像缓冲池和正式数据增强。

## 3. 数学解释

两个映射记为 `G:A→B` 与 `F:B→A`。对抗项分别要求 `G(a)` 像 B、`F(b)` 像 A；循环一致性项为 `E_a‖F(G(a))−a‖₁ + E_b‖G(F(b))−b‖₁`。代码中的 `cycle` 精确对应这两项。恒等项为 `E_b‖G(b)−b‖₁ + E_a‖F(a)−a‖₁`，帮助保留色彩或无需改变的内容。总生成器损失是对抗损失加权的 cycle 与 identity；判别器只看真实图和 detach 后的假图。循环约束并不能保证唯一正确翻译，因此不应把随机演示当成学得语义映射。
