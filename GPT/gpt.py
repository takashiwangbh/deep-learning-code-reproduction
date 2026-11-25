"""最小 GPT 式 decoder-only Transformer，包含显式因果注意力与 KV cache。

仓库根目录运行：python GPT/gpt.py。参数是随机初始化的；生成结果没有语义。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


KVCache = tuple[Tensor, Tensor]  # 每层的 (K,V)，各为 [B,H,past_time,head_size]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model 必须能被 num_heads 整除")
        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, past: KVCache | None = None) -> tuple[Tensor, KVCache]:
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.reshape(batch, time, self.num_heads, self.head_size).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        past_length = 0
        if past is not None:
            past_k, past_v = past
            past_length = past_k.size(-2)
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        # query 的绝对位置从 past_length 开始；key 从位置 0 开始。
        query_positions = past_length + torch.arange(time, device=x.device)
        key_positions = torch.arange(k.size(-2), device=x.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        context = weights @ v
        context = context.transpose(1, 2).contiguous().reshape(batch, time, channels)
        return self.output_dropout(self.projection(context)), (k, v)


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, past: KVCache | None = None) -> tuple[Tensor, KVCache]:
        attention_output, new_cache = self.attention(self.attention_norm(x), past)
        x = x + attention_output
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class GPT(nn.Module):
    """输入 token_ids [B,T]，输出每个位置预测下一个 token 的 logits。

    `use_cache=True` 时同时返回每层 K/V；后续只需传入新 token。
    学习位置嵌入用于展示 GPT 的基础结构，不含现代模型的 RoPE/MoE。
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers 至少为 1")
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(GPTBlock(d_model, num_heads, dropout) for _ in range(num_layers))
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 输出投影与输入词表共享权重，是常见的语言模型参数共享方式。
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self, token_ids: Tensor, past_key_values: list[KVCache] | None = None, use_cache: bool = False
    ) -> Tensor | tuple[Tensor, list[KVCache]]:
        if token_ids.ndim != 2 or token_ids.size(1) == 0:
            raise ValueError("token_ids 应为非空的 [batch, time] 张量")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("缓存层数应等于模型层数")
        past_length = 0 if past_key_values is None else past_key_values[0][0].size(-2)
        if past_length + token_ids.size(1) > self.block_size:
            raise ValueError("上下文长度超过 block_size")
        positions = torch.arange(
            past_length, past_length + token_ids.size(1), device=token_ids.device
        )
        x = self.dropout(self.token_embedding(token_ids) + self.position_embedding(positions))
        new_cache = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            x, cache = block(x, past)
            if use_cache:
                new_cache.append(cache)
        logits = self.lm_head(self.final_norm(x))
        return (logits, new_cache) if use_cache else logits

    @torch.no_grad()
    def generate(self, prompt: Tensor, max_new_tokens: int) -> Tensor:
        """贪心解码示例；每次只计算一个新 token 的 Q，复用旧 K/V。"""
        if prompt.ndim != 2 or prompt.size(1) == 0:
            raise ValueError("prompt 应为非空的 [batch, time] 张量")
        if max_new_tokens < 0 or prompt.size(1) + max_new_tokens > self.block_size:
            raise ValueError("生成总长度不能超过 block_size")
        if max_new_tokens == 0:
            return prompt
        result = prompt
        logits, cache = self(result, use_cache=True)
        for step in range(max_new_tokens):
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            result = torch.cat((result, next_token), dim=1)
            if step + 1 < max_new_tokens:
                logits, cache = self(next_token, past_key_values=cache, use_cache=True)
        return result


if __name__ == "__main__":
    model = GPT(vocab_size=50, block_size=16, d_model=32, num_heads=4, num_layers=2, dropout=0)
    model.eval()
    prompt = torch.tensor([[1, 2, 3]])
    print("logits:", tuple(model(prompt).shape))
    print("generated:", model.generate(prompt, max_new_tokens=3).tolist())
