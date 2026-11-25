"""Gated DeltaNet (ICLR 2025) 的核心递推与小型混合语言模型。

论文的高性能分块并行扫描、局部卷积与混合模型的滑动窗口注意力没有包含在
这个教学版本中；递推式及输出归一化/门控本身完整写出。
此文件引用同目录的可见源码 modern_decoder.py 中的 RMSNorm、SwiGLU 和 GQA。
"""

import torch
import torch.nn.functional as F
from torch import nn

from modern_decoder import GroupedQueryAttention, RMSNorm, SwiGLU


def gated_delta_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """输入 q/k/v=[B,T,H,D]，α/β=[B,T,H]；返回 [B,T,H,D]。

    Sₜ = αₜ Sₜ₋₁(I−βₜkₜkₜᵀ) + βₜvₜkₜᵀ，oₜ = Sₜqₜ。
    state 的维度 [B,H,value_dim,key_dim]；这里 value_dim=key_dim=D。
    """
    if not (q.shape == k.shape == v.shape):
        raise ValueError("q, k and v must have the same shape")
    batch, length, heads, dim = q.shape
    if alpha.shape != (batch, length, heads) or beta.shape != alpha.shape:
        raise ValueError("alpha and beta must be [batch, time, heads]")
    state = q.new_zeros(batch, heads, dim, dim)
    outputs = []
    for t in range(length):
        key = k[:, t]                            # [B,H,D]
        value = v[:, t]
        decay = alpha[:, t, :, None, None]
        write = beta[:, t, :, None]
        old_value = (state @ key[..., None]).squeeze(-1)
        # 先整体遗忘，再根据旧值修正当前 key 的关联。
        innovation = write * (value - alpha[:, t, :, None] * old_value)
        state = decay * state + innovation[..., None] * key[..., None, :]
        outputs.append((state @ q[:, t, ..., None]).squeeze(-1))
    return torch.stack(outputs, dim=1)


class GatedDeltaMixer(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.gates = nn.Linear(dim, 2 * num_heads)
        self.output_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
        self.output_gate = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = F.normalize(q.reshape(batch, length, self.num_heads, self.head_dim), dim=-1)
        k = F.normalize(k.reshape(batch, length, self.num_heads, self.head_dim), dim=-1)
        v = v.reshape(batch, length, self.num_heads, self.head_dim)
        alpha, beta = self.gates(x).sigmoid().chunk(2, dim=-1)
        output = gated_delta_scan(q, k, v, alpha, beta)
        gate = F.silu(self.output_gate(x)).reshape(batch, length, self.num_heads, self.head_dim)
        return self.output((self.output_norm(output) * gate).reshape(batch, length, dim))


class HybridBlock(nn.Module):
    """`use_attention=False` 用 Delta 递推；True 用因果 GQA。"""

    def __init__(self, dim: int, heads: int, kv_heads: int, hidden: int,
                 use_attention: bool):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.mixer = (GroupedQueryAttention(dim, heads, kv_heads) if use_attention
                      else GatedDeltaMixer(dim, heads))
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class HybridDeltaAttentionLM(nn.Module):
    """交替堆叠线性递推与 softmax 注意力；用于学习混合架构而非复制发布模型。"""

    def __init__(self, vocab_size: int, dim: int = 64, depth: int = 4,
                 heads: int = 4, kv_heads: int = 2, hidden: int = 128):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2 to show both mixers")
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            HybridBlock(dim, heads, kv_heads, hidden, use_attention=(i % 2 == 1))
            for i in range(depth)
        )
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


if __name__ == "__main__":
    model = HybridDeltaAttentionLM(32, dim=16, depth=2, heads=4, kv_heads=2, hidden=32)
    tokens = torch.randint(0, 32, (2, 6))
    logits = model(tokens)
    F.cross_entropy(logits[:, :-1].reshape(-1, 32), tokens[:, 1:].reshape(-1)).backward()
    print("hybrid logits:", tuple(logits.shape))
