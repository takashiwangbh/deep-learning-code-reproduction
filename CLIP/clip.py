"""CLIP 的双编码器与对称图文对比损失，小尺寸教学实现。

Radford et al., Learning Transferable Visual Models From Natural Language Supervision:
https://arxiv.org/abs/2103.00020

图像侧是手写 patch Transformer，文本侧是手写因果 Transformer；不导入预训练模型。
真实 CLIP 还依赖大规模配对图文、分词器和训练流程，本例只展示模型与目标函数。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, causal: bool):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads, self.head_dim, self.causal = heads, width // heads, causal
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if self.causal:
            future = torch.ones(tokens, tokens, device=x.device, dtype=torch.bool).triu(diagonal=1)
            scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        return self.projection((weights @ v).transpose(1, 2).reshape(batch, tokens, width))


class TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, causal: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = SelfAttention(width, heads, causal)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ImageEncoder(nn.Module):
    def __init__(self, image_size: int, patch_size: int, channels: int, width: int, depth: int, heads: int, projection_dim: int):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        patches = (image_size // patch_size) ** 2
        self.patch = nn.Conv2d(channels, width, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.randn(1, patches + 1, width) * 0.01)
        self.blocks = nn.Sequential(*(TransformerBlock(width, heads, False) for _ in range(depth)))
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, projection_dim, bias=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patches = self.patch(image).flatten(start_dim=2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(image.size(0), -1, -1), patches), dim=1)
        if x.size(1) != self.position.size(1):
            raise ValueError("image size must match configured size")
        x = self.blocks(x + self.position)
        return self.projection(self.norm(x[:, 0]))


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, width: int, depth: int, heads: int, projection_dim: int):
        super().__init__()
        self.context_length = context_length
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Parameter(torch.randn(1, context_length, width) * 0.01)
        self.blocks = nn.Sequential(*(TransformerBlock(width, heads, True) for _ in range(depth)))
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, projection_dim, bias=False)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if tokens.size(1) > self.context_length:
            raise ValueError("token sequence exceeds context_length")
        if torch.any(lengths < 1) or torch.any(lengths > tokens.size(1)):
            raise ValueError("lengths must be within [1, sequence length]")
        x = self.blocks(self.token(tokens) + self.position[:, :tokens.size(1)])
        # 每句的最后一个有效 token 作为句向量，通常应为 EOT。
        end = x[torch.arange(tokens.size(0), device=tokens.device), lengths - 1]
        return self.projection(self.norm(end))


class CLIP(nn.Module):
    def __init__(
        self, image_size: int = 32, patch_size: int = 8, channels: int = 3,
        vocab_size: int = 1000, context_length: int = 16, width: int = 64,
        depth: int = 2, heads: int = 4, projection_dim: int = 32,
    ):
        super().__init__()
        self.image = ImageEncoder(image_size, patch_size, channels, width, depth, heads, projection_dim)
        self.text = TextEncoder(vocab_size, context_length, width, depth, heads, projection_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, images: torch.Tensor, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        image_features = F.normalize(self.image(images), dim=-1)
        text_features = F.normalize(self.text(tokens, lengths), dim=-1)
        return self.logit_scale.exp().clamp(max=100) * image_features @ text_features.T


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """同下标图文为正样本；batch 内其他配对为负样本。"""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("logits must be a square image-text similarity matrix")
    target = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


if __name__ == "__main__":
    model = CLIP()
    images = torch.randn(3, 3, 32, 32)
    tokens = torch.randint(0, 1000, (3, 12))
    lengths = torch.tensor([12, 10, 8])
    logits = model(images, tokens, lengths)
    loss = contrastive_loss(logits)
    loss.backward()
    print(f"similarity={tuple(logits.shape)}, loss={loss.item():.3f}")
