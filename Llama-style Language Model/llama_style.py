"""LLaMA 风格的教学版解码器：RoPE + GQA + RMSNorm + SwiGLU。

这是结构学习用的小模型，不包含 tokenizer、预训练权重或完整产品模型。
所有注意力计算均在本文件中实现。仓库根目录运行：python 'Llama-style Language Model/llama_style.py'
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


KVCache = tuple[Tensor, Tensor]  # K/V: [B, num_kv_heads, past_time, head_dim]


class RMSNorm(nn.Module):
    """x / sqrt(mean(x²) + eps) * weight；不做均值中心化。"""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scaled = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return scaled.to(dtype=x.dtype) * self.weight


def apply_rope(x: Tensor, positions: Tensor, base: float = 10000.0) -> Tensor:
    """每对偶/奇通道按 token 位置旋转；x 为 [B,H,T,head_dim]。"""
    head_dim = x.size(-1)
    if head_dim % 2:
        raise ValueError("RoPE 的 head_dim 必须是偶数")
    inverse_frequency = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim)
    )
    angles = positions.float()[:, None] * inverse_frequency[None, :]
    cosine = angles.cos()[None, None, :, :].to(dtype=x.dtype)
    sine = angles.sin()[None, None, :, :].to(dtype=x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


class GroupedQueryAttention(nn.Module):
    """多组 Q 共享较少的 K/V 头，减少自回归生成时的 KV cache。"""

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        if d_model % num_heads or num_heads % num_kv_heads:
            raise ValueError("需要 d_model % num_heads == 0 且 num_heads % num_kv_heads == 0")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        if self.head_dim % 2:
            raise ValueError("RoPE 要求 head_dim 为偶数")
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, past: KVCache | None = None) -> tuple[Tensor, KVCache]:
        batch, time, _ = x.shape

        def split(tensor: Tensor, heads: int) -> Tensor:
            return tensor.reshape(batch, time, heads, self.head_dim).transpose(1, 2)

        q = split(self.q_proj(x), self.num_heads)
        k = split(self.k_proj(x), self.num_kv_heads)
        v = split(self.v_proj(x), self.num_kv_heads)
        past_length = 0 if past is None else past[0].size(-2)
        positions = past_length + torch.arange(time, device=x.device)
        q = apply_rope(q, positions)
        k = apply_rope(k, positions)
        if past is not None:
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)
        cache = (k, v)
        # 从 K 个 KV 头扩展到 H 个 Q 头；缓存仍只存 K 个头。
        repeat = self.num_heads // self.num_kv_heads
        expanded_k = k.repeat_interleave(repeat, dim=1)
        expanded_v = v.repeat_interleave(repeat, dim=1)
        scores = q @ expanded_k.transpose(-2, -1) / math.sqrt(self.head_dim)
        allowed = torch.arange(k.size(-2), device=x.device)[None, :] <= positions[:, None]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = weights @ expanded_v
        context = context.transpose(1, 2).contiguous().reshape(batch, time, -1)
        return self.o_proj(context), cache


class SwiGLU(nn.Module):
    """(SiLU(W_gate x) * W_up x) W_down。"""

    def __init__(self, d_model: int, hidden_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden_size, bias=False)
        self.up = nn.Linear(d_model, hidden_size, bias=False)
        self.down = nn.Linear(hidden_size, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class LlamaBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, hidden_size: int) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(d_model)
        self.attention = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, hidden_size)

    def forward(self, x: Tensor, past: KVCache | None = None) -> tuple[Tensor, KVCache]:
        attention_output, cache = self.attention(self.attention_norm(x), past)
        x = x + attention_output
        return x + self.ffn(self.ffn_norm(x)), cache


class LlamaStyleLM(nn.Module):
    """教学版自回归语言模型；输入 [B,T]，输出 [B,T,V]。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        num_layers: int = 2,
        hidden_size: int = 256,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers 至少为 1")
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            LlamaBlock(d_model, num_heads, num_kv_heads, hidden_size) for _ in range(num_layers)
        )
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, tokens: Tensor, past_key_values: list[KVCache] | None = None, use_cache: bool = False
    ) -> Tensor | tuple[Tensor, list[KVCache]]:
        if tokens.ndim != 2 or tokens.size(1) == 0:
            raise ValueError("tokens 应为非空的 [batch, time] 张量")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("缓存层数应等于模型层数")
        past_length = 0 if past_key_values is None else past_key_values[0][0].size(-2)
        if past_length + tokens.size(1) > self.max_length:
            raise ValueError("上下文长度超过 max_length")
        x = self.embedding(tokens)
        cache = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            x, next_cache = block(x, past)
            if use_cache:
                cache.append(next_cache)
        logits = self.output(self.norm(x))
        return (logits, cache) if use_cache else logits


if __name__ == "__main__":
    model = LlamaStyleLM(40, d_model=32, num_heads=4, num_kv_heads=2, hidden_size=64)
    print("logits:", tuple(model(torch.tensor([[1, 2, 3]])).shape))
