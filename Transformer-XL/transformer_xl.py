"""Transformer-XL 的段间记忆和相对位置注意力，显式逐段计算。"""
import math
import torch
from torch import nn


class RelativeAttention(nn.Module):
    def __init__(self, dim, heads, max_distance=128):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.width = heads, dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.r = nn.Embedding(max_distance, self.width)
        self.u = nn.Parameter(torch.zeros(heads, self.width))
        self.v = nn.Parameter(torch.zeros(heads, self.width))
        self.out = nn.Linear(dim, dim, bias=False)
        self.max_distance = max_distance

    def forward(self, x, memory=None):
        b, t, d = x.shape
        memory = x[:, :0] if memory is None else memory.detach()
        context = torch.cat([memory, x], dim=1)
        m, s = memory.size(1), context.size(1)
        q = self.q(x).reshape(b, t, self.heads, self.width).transpose(1, 2)
        k, value = self.kv(context).reshape(b, s, 2, self.heads, self.width).permute(2, 0, 3, 1, 4)
        content = (q + self.u[None, :, None]) @ k.transpose(-2, -1)
        # R_{i,j} 只依赖相对距离；缓存的旧段改变绝对下标也不会改变距离含义。
        distance = (torch.arange(t, device=x.device)[:, None] + m -
                    torch.arange(s, device=x.device)[None, :]).clamp(0, self.max_distance - 1)
        pos = self.r(distance)
        relative = torch.einsum("bhtd,tsd->bhts", q + self.v[None, :, None], pos)
        scores = (content + relative) / math.sqrt(self.width)
        allowed = torch.arange(s, device=x.device)[None, :] <= torch.arange(t, device=x.device)[:, None] + m
        attention = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min).softmax(-1)
        return self.out((attention @ value).transpose(1, 2).reshape(b, t, d))


class XLBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = RelativeAttention(dim, heads)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.ReLU(), nn.Linear(4 * dim, dim))

    def forward(self, x, memory=None):
        x = self.norm1(x + self.attn(x, memory))
        return self.norm2(x + self.ff(x))


class TransformerXL(nn.Module):
    def __init__(self, vocab=128, dim=64, heads=4, layers=2, mem_len=16):
        super().__init__()
        if mem_len < 1:
            raise ValueError("mem_len 至少为 1")
        self.embedding = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(XLBlock(dim, heads) for _ in range(layers))
        self.head = nn.Linear(dim, vocab)
        self.mem_len = mem_len

    def forward(self, ids, memories=None):
        x = self.embedding(ids)
        new_memories = []
        memories = [None] * len(self.layers) if memories is None else memories
        for layer, old in zip(self.layers, memories):
            new_memories.append(torch.cat([old, x], 1)[:, -self.mem_len:].detach() if old is not None
                                else x[:, -self.mem_len:].detach())
            x = layer(x, old)
        return self.head(x), new_memories


if __name__ == "__main__":
    model = TransformerXL(vocab=32, dim=32, layers=2, mem_len=3)
    first, mem = model(torch.randint(0, 32, (2, 4)))
    second, mem = model(torch.randint(0, 32, (2, 3)), mem)
    second.sum().backward()
    print("segment logits", tuple(first.shape), tuple(second.shape), "memory", tuple(mem[0].shape))
