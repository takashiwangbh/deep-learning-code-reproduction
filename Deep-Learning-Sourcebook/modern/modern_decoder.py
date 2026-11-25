"""RoPE + GQA + RMSNorm + SwiGLU + 稀疏 MoE 的小型自回归语言模型。

这是逐项展示现代 LLM 组件的教学模型，不对应某个商用模型的完整配置。
运行 `python modern_decoder.py` 可做一次前向/反向传播。
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """对每一对偶/奇维度做二维旋转；x 是 [B, H, T, D]。"""
    dim = x.size(-1)
    if dim % 2:
        raise ValueError("RoPE head dimension must be even")
    inv_freq = base ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    angles = positions.float()[:, None] * inv_freq[None, :]  # [T, D/2]
    cos = angles.cos().repeat_interleave(2, dim=-1).to(x.dtype)[None, None]
    sin = angles.sin().repeat_interleave(2, dim=-1).to(x.dtype)[None, None]
    pairs = x.reshape(*x.shape[:-1], dim // 2, 2)
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    return x * cos + rotated * sin


class GroupedQueryAttention(nn.Module):
    """H 个 query head 共享 K 个 key/value head；K<=H 且 H 能被 K 整除。"""

    def __init__(self, dim: int, query_heads: int, kv_heads: int):
        super().__init__()
        if dim % query_heads or query_heads % kv_heads or (dim // query_heads) % 2:
            raise ValueError("dim/query_heads must be even and query_heads divisible by kv_heads")
        self.query_heads, self.kv_heads = query_heads, kv_heads
        self.head_dim = dim // query_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _split(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        b, t, _ = x.shape
        return x.reshape(b, t, heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        positions = torch.arange(t, device=x.device)
        q = apply_rope(self._split(self.q_proj(x), self.query_heads), positions)
        k = apply_rope(self._split(self.k_proj(x), self.kv_heads), positions)
        v = self._split(self.v_proj(x), self.kv_heads)
        # 相邻 query head 共用同一组 K/V，参数量与 KV cache 均减少。
        repeat = self.query_heads // self.kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        future = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(diagonal=1)
        scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        context = (weights @ v).transpose(1, 2).reshape(b, t, -1)
        return self.out_proj(context)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SparseMoE(nn.Module):
    """逐 token top-k 路由的易读实现；返回输出和辅助负载均衡损失。"""

    def __init__(self, dim: int, hidden: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(dim, hidden) for _ in range(num_experts))
        self.num_experts, self.top_k = num_experts, top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        flat = x.reshape(-1, x.size(-1))
        router_logits = self.router(flat)                 # [B*T, E]
        router_prob = router_logits.softmax(dim=-1)
        top_logits, chosen = router_logits.topk(self.top_k, dim=-1)
        gates = top_logits.softmax(dim=-1)               # 选中专家的权重加和为 1
        result = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            token_index, slot = (chosen == expert_index).nonzero(as_tuple=True)
            if token_index.numel():
                contribution = expert(flat[token_index]) * gates[token_index, slot, None]
                result = result.index_add(0, token_index, contribution)
        # f_i 不可导，P_i 可导。鼓励分配率与平均路由概率都不过度集中。
        assignment = F.one_hot(chosen, self.num_experts).float().mean(dim=(0, 1))
        average_probability = router_prob.mean(dim=0)
        auxiliary_loss = self.num_experts * (assignment * average_probability).sum()
        return result.reshape(original_shape), auxiliary_loss


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, query_heads: int, kv_heads: int, hidden: int,
                 num_experts: int = 0):
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.attention = GroupedQueryAttention(dim, query_heads, kv_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SparseMoE(dim, hidden, num_experts) if num_experts else SwiGLU(dim, hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attention(self.attention_norm(x))
        if isinstance(self.ffn, SparseMoE):
            feedforward, auxiliary_loss = self.ffn(self.ffn_norm(x))
        else:
            feedforward = self.ffn(self.ffn_norm(x))
            auxiliary_loss = x.new_zeros(())
        return x + feedforward, auxiliary_loss


class ModernDecoderLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128, depth: int = 4,
                 query_heads: int = 8, kv_heads: int = 2, hidden: int = 256,
                 num_experts: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            DecoderBlock(dim, query_heads, kv_heads, hidden, num_experts)
            for _ in range(depth)
        )
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight        # 输入/输出词嵌入共享参数

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(tokens)
        auxiliary_losses = []
        for block in self.blocks:
            x, auxiliary_loss = block(x)
            auxiliary_losses.append(auxiliary_loss)
        logits = self.lm_head(self.final_norm(x))         # [B, T, vocab_size]
        return logits, torch.stack(auxiliary_losses).mean()


if __name__ == "__main__":
    model = ModernDecoderLM(vocab_size=32, dim=32, depth=2, query_heads=4,
                            kv_heads=2, hidden=64, num_experts=4)
    tokens = torch.randint(0, 32, (2, 8))
    logits, aux = model(tokens)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 32), tokens[:, 1:].reshape(-1))
    (loss + 0.01 * aux).backward()
    print("logits:", tuple(logits.shape), "language loss:", round(loss.item(), 4))
