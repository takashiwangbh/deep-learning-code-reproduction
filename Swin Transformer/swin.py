"""Swin Transformer：非重叠窗口、移位窗口与 patch merging。"""
import math
import torch
from torch import nn


def partition(x, window):
    b, h, w, c = x.shape
    if h % window or w % window:
        raise ValueError("H、W 必须是窗口尺寸的整数倍")
    return x.reshape(b, h // window, window, w // window, window, c).permute(0, 1, 3, 2, 4, 5).reshape(
        b, h // window * (w // window), window * window, c
    )


def reverse(windows, h, w, window):
    b, _, _, c = windows.shape
    return windows.reshape(b, h // window, w // window, window, window, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)


def shifted_mask(h, w, window, shift, device):
    """九块区域编号；阻止 cyclic roll 把图片左右/上下边缘错误连起来。"""
    region = torch.zeros(1, h, w, 1, device=device)
    spans = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
    count = 0
    for rows in spans:
        for cols in spans:
            region[:, rows, cols] = count
            count += 1
    ids = partition(region, window)[0, :, :, 0]
    return ids[:, :, None] != ids[:, None, :]


class WindowAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.width = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, windows, mask=None):
        b, nw, n, d = windows.shape
        q, k, v = self.qkv(windows).reshape(b, nw, n, 3, self.heads, self.width).permute(3, 0, 1, 4, 2, 5)
        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.width)
        if mask is not None:
            score = score.masked_fill(mask[None, :, None], torch.finfo(score.dtype).min)
        weight = score.softmax(-1)
        return self.out((weight @ v).transpose(2, 3).reshape(b, nw, n, d))


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, window, shift=0):
        super().__init__()
        self.window, self.shift = window, shift
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        b, h, w, c = x.shape
        shortcut = x
        y = self.norm1(x)
        # 单个窗口已经覆盖整张阶段特征图；此时不做平移。
        shift = self.shift if h > self.window and w > self.window else 0
        if shift:
            y = torch.roll(y, shifts=(-shift, -shift), dims=(1, 2))
        mask = shifted_mask(h, w, self.window, shift, x.device) if shift else None
        y = reverse(self.attn(partition(y, self.window), mask), h, w, self.window)
        if shift:
            y = torch.roll(y, shifts=(shift, shift), dims=(1, 2))
        x = shortcut + y
        return x + self.ff(self.norm2(x))


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduce = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        b, h, w, c = x.shape
        if h % 2 or w % 2:
            raise ValueError("合并前宽高必须为偶数")
        group = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.reduce(self.norm(group))


class SwinTransformer(nn.Module):
    def __init__(self, dim=32, window=4, classes=10):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, 4, stride=4)
        self.stage1 = nn.Sequential(SwinBlock(dim, 4, window), SwinBlock(dim, 4, window, shift=window // 2))
        self.merge = PatchMerging(dim)
        self.stage2 = nn.Sequential(SwinBlock(2 * dim, 4, window), SwinBlock(2 * dim, 4, window, shift=window // 2))
        self.norm = nn.LayerNorm(2 * dim)
        self.head = nn.Linear(2 * dim, classes)

    def forward(self, image):
        x = self.patch(image).permute(0, 2, 3, 1)
        x = self.stage1(x)
        x = self.stage2(self.merge(x))
        return self.head(self.norm(x).mean((1, 2)))


if __name__ == "__main__":
    model = SwinTransformer(dim=16)
    logits = model(torch.randn(2, 3, 32, 32))
    logits.sum().backward()
    print("Swin", tuple(logits.shape))
