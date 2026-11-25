# 04 · 现代模型组件

先读 `sequence/` 的 Transformer 和 GPT，再看这些源码。这里实现的是**可组合的算法机制**，并不声称重现某个大型模型的全部训练流程、权重或性能。

| 源码 | 关键机制 | 形状 |
| --- | --- | --- |
| `modern_decoder.py` | RoPE、分组查询注意力 GQA、RMSNorm、SwiGLU、top-k MoE | token `[B,T]` → logits `[B,T,V]` |
| `mamba_selective_ssm.py` | 输入相关的 Δ/B/C、因果深度卷积、选择性状态递推 | token `[B,T]` → logits `[B,T,V]` |
| `gated_delta.py` | 2025 年 Gated DeltaNet 的遗忘门与定向覆写；交替堆叠线性递推/GQA | token `[B,T]` → logits `[B,T,V]` |
| `gated_delta2.py` | 2026 年 Gated DeltaNet-2 的逐通道衰减、独立擦除门与写入门 | token `[B,T]` → logits `[B,T,V]` |
| `dpo.py` | 比较偏好回答与拒绝回答相对于参考模型的 log 概率差 | 四个 `[B]` → 标量损失 |

从仓库根目录运行：

```bash
python Deep-Learning-Sourcebook/modern/modern_decoder.py
python Deep-Learning-Sourcebook/modern/mamba_selective_ssm.py
python Deep-Learning-Sourcebook/modern/gated_delta.py
python Deep-Learning-Sourcebook/modern/gated_delta2.py
python Deep-Learning-Sourcebook/modern/dpo.py
python Deep-Learning-Sourcebook/modern/smoke_test.py
```

## 公式与源码的对应

- **RoPE**：`apply_rope` 对相邻两个 head 维度作旋转，角度由 token 位置产生；进入注意力点积前分别旋转 Q/K。
- **GQA**：`GroupedQueryAttention` 产生 H 组 Q、K 组 K/V；`repeat_interleave` 让多组 Q 共享同一 K/V。未来位置在 softmax 前被遮住。
- **MoE**：`SparseMoE` 给每个 token 选择 top-k 专家，按门控权重求和；辅助损失用于减轻专家分配不均。这里没有分布式 token dispatch 或专家容量限制。
- **选择性 SSM**：`MambaBlock` 显式计算 `hₜ=exp(ΔₜA)hₜ₋₁+ΔₜBₜuₜ` 与 `yₜ=Cₜhₜ+Duₜ`。这一版是便于阅读的逐步扫描，速度不能代表原论文优化实现。
- **Gated DeltaNet**：`gated_delta_scan` 显式计算 `Sₜ=αₜSₜ₋₁(I−βₜkₜkₜᵀ)+βₜvₜkₜᵀ`；`α` 遗忘、`β` 定向写入。输出随后经过逐 head 归一化、SiLU 门控与线性投影。`HybridDeltaAttentionLM` 交替使用该递推和因果 GQA，展示混合模型的组织方式；教学版未实现论文的并行扫描、局部卷积或滑动窗口注意力。
- **Gated DeltaNet-2**：`gated_delta2_scan` 把单个 `β` 拆成 key 通道的擦除门 `b` 与 value 通道的写入门 `w`；设置 `b=w=β` 且所有通道共享 `α` 时，退化为上一行的递推。教学版使用顺序扫描，省略并行核、短卷积、分组 value head 和滑动窗口注意力。
- **DPO**：`dpo_loss` 实现 `−log σ(β[(logπθ(y⁺)−logπθ(y⁻))−(logπref(y⁺)−logπref(y⁻))])`；`sequence_log_probability` 负责从逐 token logits 汇总回答概率。

## 原始论文

- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- [GQA](https://arxiv.org/abs/2305.13245)
- [RMSNorm](https://arxiv.org/abs/1910.07467)
- [GLU Variants / SwiGLU](https://arxiv.org/abs/2002.05202)
- [Sparsely-Gated Mixture-of-Experts / 稀疏 top-k 专家](https://arxiv.org/abs/1701.06538)
- [Switch Transformer / top-1 路由与负载均衡背景](https://arxiv.org/abs/2101.03961)
- [Mamba / 选择性状态空间](https://arxiv.org/abs/2312.00752)
- [Gated Delta Networks / ICLR 2025](https://arxiv.org/abs/2412.06464)
- [Gated DeltaNet-2 / 2026](https://arxiv.org/abs/2605.22791)
- [DPO](https://arxiv.org/abs/2305.18290)

论文与官方仓库仅用来确认结构和公式；此目录源码为本仓库重新编写的教学实现，没有复制外部仓库文件。
