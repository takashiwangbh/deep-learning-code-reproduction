"""DiT 核心：把图像或潜变量切成 patch，用时间条件调制 Transformer。

Peebles & Xie, Scalable Diffusion Models with Transformers:
https://arxiv.org/abs/2212.09748

这是噪声预测网络 epsilon_theta(x_t,t[,y])，可传给 ddpm.DDPM；不包含原论文的
大规模 VAE 编码器、训练系统或 classifier-free guidance。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequency = torch.exp(
        -math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    phase = t.float()[:, None] * frequency[None]
    return F.pad(torch.cat((phase.cos(), phase.sin()), dim=1), (0, dim % 2))


def position_embedding_2d(grid_size: int, dim: int) -> torch.Tensor:
    """固定二维 sin/cos 位置编码，返回 [1, grid_size², dim]。"""
    if dim % 4:
        raise ValueError("hidden_dim must be divisible by 4")
    axis_dim = dim // 4
    frequency = torch.exp(-math.log(10_000) * torch.arange(axis_dim, dtype=torch.float32) / max(axis_dim - 1, 1))
    coordinates = torch.arange(grid_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    y_phase = yy.reshape(-1, 1) * frequency[None]
    x_phase = xx.reshape(-1, 1) * frequency[None]
    return torch.cat((y_phase.sin(), y_phase.cos(), x_phase.sin(), x_phase.cos()), dim=1)[None]


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, hidden = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        weights = scores.softmax(dim=-1)
        attended = (weights @ v).transpose(1, 2).reshape(batch, tokens, hidden)
        return self.projection(attended)


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attention = SelfAttention(hidden_dim, heads)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio), nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.condition(c).chunk(6, dim=1)
        x = x + gate_a[:, None, :] * self.attention(modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_m[:, None, :] * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.condition(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DiT(nn.Module):
    def __init__(
        self, image_size: int = 32, patch_size: int = 4, in_channels: int = 4,
        hidden_dim: int = 256, depth: int = 4, heads: int = 4, num_classes: int | None = None,
    ):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, patch_size, stride=patch_size)
        self.register_buffer("pos_embed", position_embedding_2d(image_size // patch_size, hidden_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(256, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.class_embed = nn.Embedding(num_classes, hidden_dim) if num_classes is not None else None
        self.blocks = nn.ModuleList(DiTBlock(hidden_dim, heads) for _ in range(depth))
        self.final = FinalLayer(hidden_dim, patch_size, in_channels)
        # adaLN-Zero：起始时每个残差分支为 0，最终噪声预测也为 0。
        for block in self.blocks:
            nn.init.zeros_(block.condition[-1].weight)
            nn.init.zeros_(block.condition[-1].bias)
        nn.init.zeros_(self.final.condition[-1].weight)
        nn.init.zeros_(self.final.condition[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[1:] != (self.in_channels, self.image_size, self.image_size):
            raise ValueError("input shape must match configured channels and image_size")
        if self.class_embed is not None and labels is None:
            raise ValueError("labels required for a class-conditional DiT")
        h = self.patch_embed(x).flatten(start_dim=2).transpose(1, 2) + self.pos_embed
        c = self.time_mlp(timestep_embedding(t, 256))
        if self.class_embed is not None:
            c = c + self.class_embed(labels)
        for block in self.blocks:
            h = block(h, c)
        patches = self.final(h, c)
        grid = self.image_size // self.patch_size
        patches = patches.reshape(x.size(0), grid, grid, self.patch_size, self.patch_size, self.in_channels)
        return patches.permute(0, 5, 1, 3, 2, 4).reshape_as(x)


if __name__ == "__main__":
    model = DiT(image_size=8, patch_size=2, in_channels=2, hidden_dim=32, depth=2, heads=4)
    output = model(torch.randn(2, 2, 8, 8), torch.tensor([1, 5]))
    F.mse_loss(output, torch.randn_like(output)).backward()
    print(f"predicted_noise={tuple(output.shape)}")
