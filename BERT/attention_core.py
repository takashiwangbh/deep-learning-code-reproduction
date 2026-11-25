"""BERT 使用的多头注意力，完整张量运算保留在本项目内。"""

import math

import torch
from torch import Tensor, nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model 必须能被 num_heads 整除")
        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, time, _ = x.shape
        return x.reshape(batch, time, self.num_heads, self.head_size).transpose(1, 2)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                mask: Tensor | None = None) -> Tensor:
        q = self._split_heads(self.query(query))
        k = self._split_heads(self.key(key))
        v = self._split_heads(self.value(value))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        if mask is not None:
            weights = weights.masked_fill(~mask, 0.0)
        context = self.dropout(weights) @ v
        context = context.transpose(1, 2).contiguous().reshape(query.size(0), query.size(1), -1)
        return self.output(context)
