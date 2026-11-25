"""Perceiver：大量输入 token 通过交叉注意力汇入固定数量潜变量。"""
import math
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.width = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, query, context):
        b, nq, d = query.shape
        nk = context.size(1)
        q = self.q(query).reshape(b, nq, self.heads, self.width).transpose(1, 2)
        k = self.k(context).reshape(b, nk, self.heads, self.width).transpose(1, 2)
        v = self.v(context).reshape(b, nk, self.heads, self.width).transpose(1, 2)
        weight = ((q @ k.transpose(-2, -1)) / math.sqrt(self.width)).softmax(-1)
        return self.out((weight @ v).transpose(1, 2).reshape(b, nq, d))


class Perceiver(nn.Module):
    def __init__(self, input_dim=3, dim=64, latents=16, depth=2, classes=10):
        super().__init__()
        self.input_project = nn.Linear(input_dim + 6, dim)
        self.latents = nn.Parameter(torch.randn(1, latents, dim) * 0.02)
        self.cross = nn.ModuleList(Attention(dim) for _ in range(depth))
        self.self_attention = nn.ModuleList(Attention(dim) for _ in range(depth))
        self.ff = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, inputs):
        # inputs [B,N,C]；简单傅里叶位置特征展示原论文的坐标信息思想。
        b, n, _ = inputs.shape
        coord = torch.linspace(-1, 1, n, device=inputs.device, dtype=inputs.dtype)
        freq = torch.tensor([1., 2., 4.], device=inputs.device, dtype=inputs.dtype)
        angle = math.pi * coord[:, None] * freq
        position = torch.cat([angle.sin(), angle.cos()], -1).expand(b, -1, -1)
        context = self.input_project(torch.cat([inputs, position], -1))
        latent = self.latents.expand(b, -1, -1)
        for cross, self_attn, ff in zip(self.cross, self.self_attention, self.ff):
            latent = latent + cross(self.norm(latent), context)
            latent = latent + self_attn(self.norm(latent), self.norm(latent))
            latent = latent + ff(latent)
        return self.head(self.norm(latent).mean(1))


if __name__ == "__main__":
    model = Perceiver(input_dim=3, dim=32, latents=8)
    y = model(torch.randn(2, 40, 3))
    y.sum().backward()
    print("Perceiver", tuple(y.shape))
