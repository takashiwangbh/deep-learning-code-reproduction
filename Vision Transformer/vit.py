"""Vision Transformer：把图像块当作 token，用显式实现的自注意力分类。"""

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """QK^T / sqrt(d_head) → softmax → V；不依赖封装好的注意力层。"""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim 必须能被 num_heads 整除")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        output = self.attn_drop(attention) @ v
        output = output.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(output))


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))  # 第一条残差路径
        return x + self.mlp(self.norm2(x))  # 第二条残差路径


class VisionTransformer(nn.Module):
    """默认 ViT-B/16 结构：224 图像、16 像素块、12 层、12 头、768 维。"""

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 num_classes: int = 1000, dim: int = 768,
                 depth: int = 12, num_heads: int = 12, dropout: float = 0.0):
        super().__init__()
        if img_size <= 0 or patch_size <= 0 or img_size % patch_size:
            raise ValueError("img_size 必须是 patch_size 的正整数倍")
        if depth <= 0:
            raise ValueError("depth 必须为正")
        num_patches = (img_size // patch_size) ** 2
        # stride=patch_size 的卷积等价于逐块展平后共享的线性投影。
        self.patch_embed = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*(EncoderBlock(dim, num_heads, dropout=dropout)
                                      for _ in range(depth)))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.img_size = img_size
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (3, self.img_size, self.img_size):
            raise ValueError(f"VisionTransformer 需要 [B, 3, {self.img_size}, {self.img_size}] 输入")
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = self.pos_drop(torch.cat([cls, patches], dim=1) + self.pos_embed)
        return self.head(self.norm(self.blocks(tokens))[:, 0])


if __name__ == "__main__":
    model = VisionTransformer(img_size=32, patch_size=8, dim=32,
                              depth=2, num_heads=4, num_classes=10)
    print(model(torch.randn(2, 3, 32, 32)).shape)
