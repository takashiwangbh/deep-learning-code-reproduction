"""Longformer 的局部窗口 + 全局 token 注意力；按稀疏邻居计算。"""
import math
import torch
from torch import nn


class LocalGlobalAttention(nn.Module):
    def __init__(self, dim=64, heads=4, window=2):
        super().__init__()
        assert dim % heads == 0 and window >= 0
        self.heads, self.width, self.window = heads, dim // heads, window
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, global_mask=None):
        b, t, d = x.shape
        if global_mask is None:
            global_mask = torch.zeros(t, dtype=torch.bool, device=x.device)
            global_mask[0] = True
        if global_mask.shape != (t,):
            raise ValueError("global_mask 需要形状 [T]，同一批样本共用位置")
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, self.width).permute(2, 0, 3, 1, 4)
        outputs = []
        all_positions = torch.arange(t, device=x.device)
        for i in range(t):
            if bool(global_mask[i]):
                index = all_positions
            else:
                index = all_positions[(all_positions - i).abs().le(self.window) | global_mask]
            score = torch.einsum("bhd,bhsd->bhs", q[:, :, i], k[:, :, index]) / math.sqrt(self.width)
            weights = score.softmax(-1)
            outputs.append(torch.einsum("bhs,bhsd->bhd", weights, v[:, :, index]))
        y = torch.stack(outputs, dim=2).transpose(1, 2).reshape(b, t, d)
        return self.out(y)


class Longformer(nn.Module):
    def __init__(self, vocab=128, dim=64, heads=4, window=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.attn = LocalGlobalAttention(dim, heads, window)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids, global_mask=None):
        x = self.embed(ids)
        x = self.norm1(x + self.attn(x, global_mask))
        return self.head(self.norm2(x + self.ff(x)))


if __name__ == "__main__":
    model = Longformer(vocab=32, dim=32, heads=4, window=1)
    logits = model(torch.randint(0, 32, (2, 7)))
    logits.sum().backward()
    print("Longformer", tuple(logits.shape))
