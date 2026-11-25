"""Linformer：在序列长度维投影键和值，而不是缩小通道宽度。"""
import math
import torch
from torch import nn


class LinformerAttention(nn.Module):
    def __init__(self, dim=64, heads=4, max_length=128, rank=16):
        super().__init__()
        assert dim % heads == 0 and 0 < rank <= max_length
        self.heads, self.width, self.max_length = heads, dim // heads, max_length
        self.q, self.k, self.v, self.out = [nn.Linear(dim, dim) for _ in range(4)]
        self.E = nn.Parameter(torch.randn(heads, rank, max_length) / math.sqrt(max_length))
        self.F = nn.Parameter(torch.randn(heads, rank, max_length) / math.sqrt(max_length))

    def forward(self, x):
        b, n, d = x.shape
        if n > self.max_length:
            raise ValueError("长度超过 max_length")
        split = lambda z: z.reshape(b, n, self.heads, self.width).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        # [H,R,N] × [B,H,N,D] -> [B,H,R,D]。显式写出长度维压缩。
        k_short = torch.einsum("hrn,bhnd->bhrd", self.E[:, :, :n], k)
        v_short = torch.einsum("hrn,bhnd->bhrd", self.F[:, :, :n], v)
        attention = ((q @ k_short.transpose(-2, -1)) / math.sqrt(self.width)).softmax(-1)
        return self.out((attention @ v_short).transpose(1, 2).reshape(b, n, d))


class Linformer(nn.Module):
    def __init__(self, vocab=128, dim=64, heads=4, max_length=128, rank=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.attn = LinformerAttention(dim, heads, max_length, rank)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids):
        x = self.embed(ids)
        return self.head(self.norm(x + self.attn(x)))


if __name__ == "__main__":
    model = Linformer(vocab=32, dim=32, max_length=8, rank=3)
    logits = model(torch.randint(0, 32, (2, 8)))
    logits.sum().backward()
    print("Linformer", tuple(logits.shape))
