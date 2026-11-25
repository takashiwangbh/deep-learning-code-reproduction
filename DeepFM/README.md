# DeepFM：共享嵌入的低阶与高阶特征交互

## 1. 出处、提出者与意义

Huifeng Guo、Ruiming Tang、Yunming Ye、Zhenguo Li、Xiuqiang He 在 2017 年提出 [《DeepFM: A Factorization-Machine based Neural Network for CTR Prediction》](https://arxiv.org/abs/1703.04247)。点击率预测需要发现类别特征之间的组合：两个字段一起出现可能很重要，三个及更多字段也可能互相影响。DeepFM 让因子分解机（FM）负责一阶和二阶交互，让 MLP 学习更复杂的交互，而且两者共享特征嵌入。与手工指定大量交叉项相比，这提供了一条更自动的路径。

## 2. 源码导航与直观解释

[deepfm.py](deepfm.py) 接收每个字段一个类别 ID，输入形状 `[B,F]`。`offsets` 把不同字段的局部 ID 放进互不重叠的全局 ID 区间；`first_order` 给出单个特征的权重，`embedding` 给出每个特征的向量。`forward` 同时计算一阶和、FM 二阶项与把嵌入展平后的 MLP 分数，最终返回 `[B]` 点击 logit。根目录运行 `python DeepFM/deepfm.py`，可看到小批次反向传播。示例没有真实广告日志、缺失字段处理或论文评测。

## 3. 数学解释

把第 `i` 个活跃特征的权重和嵌入记为 `wᵢ,vᵢ`。FM 部分是 `z_FM=Σ_i wᵢ + Σ_{i<j}〈vᵢ,vⱼ〉`，即每对字段向量的内积。直接枚举每一对要二重循环；利用恒等式 `Σ_{i<j}〈vᵢ,vⱼ〉 = 1/2 Σ_k[(Σ_i vᵢk)²-Σ_i vᵢk²]`，源码只需求和、平方即可计算。Deep 分支 `z_deep=MLP([v₁;…;v_F])`，并使用同一个 `embedding` 参数。最终 `P(click)=sigmoid(b+z_FM+z_deep)`；源码返回括号里的 logit 给稳定的二元交叉熵函数。
