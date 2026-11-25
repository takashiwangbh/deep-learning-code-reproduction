"""Switch Transformer：每个 token 只路由至一个 FFN 专家。"""
import math
import torch
from torch import nn


class SwitchFFN(nn.Module):
    def __init__(self, dim=64, experts=4, capacity_factor=1.25):
        super().__init__()
        self.router = nn.Linear(dim, experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, 4 * dim), nn.ReLU(), nn.Linear(4 * dim, dim))
            for _ in range(experts)
        )
        self.capacity_factor = capacity_factor

    def forward(self, x):
        b, t, d = x.shape
        flat = x.reshape(b * t, d)
        probability = self.router(flat.float()).softmax(-1)
        choice = probability.argmax(-1)
        capacity = max(1, math.ceil(self.capacity_factor * b * t / len(self.experts)))
        output = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            index = torch.nonzero(choice == i, as_tuple=False).flatten()[:capacity]
            if index.numel():
                expert_output = expert(flat[index]) * probability[index, i:i+1].to(flat.dtype)
                output = output.index_copy(0, index, expert_output)
        fraction = torch.stack([(choice == i).float().mean() for i in range(len(self.experts))])
        mean_probability = probability.mean(0)
        auxiliary = len(self.experts) * (fraction * mean_probability).sum()
        return output.reshape(b, t, d), auxiliary


class SelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.width = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, self.width).permute(2, 0, 3, 1, 4)
        weight = ((q @ k.transpose(-2, -1)) / math.sqrt(self.width)).softmax(-1)
        return self.out((weight @ v).transpose(1, 2).reshape(b, t, d))


class SwitchTransformer(nn.Module):
    def __init__(self, vocab=128, dim=64, heads=4, experts=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.attn = SelfAttention(dim, heads)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.moe = SwitchFFN(dim, experts)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids):
        x = self.embed(ids)
        attended = self.attn(self.norm1(x))
        x = x + attended
        routed, auxiliary = self.moe(self.norm2(x))
        return self.head(x + routed), auxiliary


if __name__ == "__main__":
    model = SwitchTransformer(vocab=32, dim=32, experts=4)
    logits, aux = model(torch.randint(0, 32, (2, 5)))
    (logits.square().mean() + 0.01 * aux).backward()
    print("Switch logits", tuple(logits.shape), "aux", round(aux.item(), 4))
