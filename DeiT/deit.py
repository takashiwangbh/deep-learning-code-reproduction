"""DeiT：带蒸馏 token 的视觉 Transformer 与硬标签蒸馏损失。"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.width = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.width).permute(2, 0, 3, 1, 4)
        weight = ((q @ k.transpose(-2, -1)) / math.sqrt(self.width)).softmax(-1)
        return self.out((weight @ v).transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class DeiT(nn.Module):
    def __init__(self, image_size=32, patch=8, dim=64, heads=4, depth=2, classes=10):
        super().__init__()
        assert image_size % patch == 0
        tokens = (image_size // patch) ** 2
        self.patch = nn.Conv2d(3, dim, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.distill = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, tokens + 2, dim))
        self.blocks = nn.Sequential(*(Block(dim, heads) for _ in range(depth)))
        self.norm = nn.LayerNorm(dim)
        self.class_head = nn.Linear(dim, classes)
        self.distill_head = nn.Linear(dim, classes)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, image):
        b = image.size(0)
        patches = self.patch(image).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(b, -1, -1), self.distill.expand(b, -1, -1), patches], 1)
        if x.size(1) != self.position.size(1):
            raise ValueError("图像尺寸与 image_size 不一致")
        x = self.norm(self.blocks(x + self.position))
        return self.class_head(x[:, 0]), self.distill_head(x[:, 1])


def hard_distillation_loss(class_logits, distill_logits, labels, teacher_logits, alpha=0.5):
    """原论文硬蒸馏：学生分类头学真实标签，蒸馏头学教师最大概率类。"""
    teacher_label = teacher_logits.detach().argmax(-1)
    return (1 - alpha) * F.cross_entropy(class_logits, labels) + alpha * F.cross_entropy(distill_logits, teacher_label)


if __name__ == "__main__":
    model = DeiT()
    ordinary, distilled = model(torch.randn(2, 3, 32, 32))
    loss = hard_distillation_loss(ordinary, distilled, torch.tensor([1, 2]), torch.randn(2, 10))
    loss.backward()
    print("DeiT heads", tuple(ordinary.shape), tuple(distilled.shape), "averaged", tuple(((ordinary + distilled) / 2).shape))
