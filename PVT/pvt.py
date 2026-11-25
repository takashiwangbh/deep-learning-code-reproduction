"""PVT：四级金字塔 patch 表示与空间缩减注意力。"""
import math
import torch
from torch import nn


class SpatialReductionAttention(nn.Module):
    def __init__(self, dim, heads, reduction):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.width, self.reduction = heads, dim // heads, reduction
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.sr = nn.Conv2d(dim, dim, reduction, stride=reduction) if reduction > 1 else None
        self.sr_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, h, w):
        b, n, d = x.shape
        q = self.q(x).reshape(b, n, self.heads, self.width).transpose(1, 2)
        if self.sr is not None:
            reduced = self.sr(x.transpose(1, 2).reshape(b, d, h, w)).flatten(2).transpose(1, 2)
        else:
            reduced = x
        s = reduced.size(1)
        k, v = self.kv(self.sr_norm(reduced)).reshape(b, s, 2, self.heads, self.width).permute(2, 0, 3, 1, 4)
        attention = ((q @ k.transpose(-2, -1)) / math.sqrt(self.width)).softmax(-1)
        return self.out((attention @ v).transpose(1, 2).reshape(b, n, d))


class PVTBlock(nn.Module):
    def __init__(self, dim, heads, reduction):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(dim, heads, reduction)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, h, w):
        x = x + self.attn(self.norm1(x), h, w)
        return x + self.ff(self.norm2(x))


class PVT(nn.Module):
    def __init__(self, dim=16, classes=10):
        super().__init__()
        widths = [dim, dim * 2, dim * 4, dim * 8]
        strides = [4, 2, 2, 2]
        reductions = [8, 4, 2, 1]
        self.patch_layers = nn.ModuleList()
        self.blocks = nn.ModuleList()
        previous = 3
        for width, stride, reduction in zip(widths, strides, reductions):
            self.patch_layers.append(nn.Conv2d(previous, width, stride, stride=stride))
            self.blocks.append(PVTBlock(width, heads=4, reduction=reduction))
            previous = width
        self.head = nn.Linear(widths[-1], classes)

    def forward_features(self, image):
        features = []
        x = image
        for patch, block in zip(self.patch_layers, self.blocks):
            x = patch(x)
            b, c, h, w = x.shape
            tokens = block(x.flatten(2).transpose(1, 2), h, w)
            x = tokens.transpose(1, 2).reshape(b, c, h, w)
            features.append(x)
        return features

    def forward(self, image):
        return self.head(self.forward_features(image)[-1].mean((2, 3)))


if __name__ == "__main__":
    model = PVT(dim=8)
    image = torch.randn(2, 3, 64, 64)
    features = model.forward_features(image)
    logits = model(image)
    logits.sum().backward()
    print("PVT pyramid", [tuple(f.shape) for f in features], "logits", tuple(logits.shape))
