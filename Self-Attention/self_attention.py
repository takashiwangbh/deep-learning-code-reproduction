"""Scaled dot-product and multi-head self-attention with visible masking."""

import math
import torch
from torch import nn


def scaled_dot_product(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, allowed=None):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if allowed is not None:
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, causal: bool = False):
        batch, steps, width = x.shape
        q, k, v = self.qkv(x).reshape(batch, steps, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4)
        mask = torch.ones(steps, steps, device=x.device, dtype=torch.bool).tril() if causal else None
        context, weights = scaled_dot_product(q, k, v, mask)
        context = context.transpose(1, 2).reshape(batch, steps, width)
        return self.output(context), weights


if __name__ == "__main__":
    model = MultiHeadSelfAttention(16, 4)
    y, attention = model(torch.randn(2, 5, 16), causal=True)
    y.sum().backward()
    print("output:", tuple(y.shape), "attention:", tuple(attention.shape))
