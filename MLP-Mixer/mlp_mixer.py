"""MLP-Mixer：分别沿 token 和通道两个轴混合。"""
import torch
from torch import nn


class MixerBlock(nn.Module):
    def __init__(self, tokens, dim, token_hidden, channel_hidden):
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(nn.Linear(tokens, token_hidden), nn.GELU(), nn.Linear(token_hidden, tokens))
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(nn.Linear(dim, channel_hidden), nn.GELU(), nn.Linear(channel_hidden, dim))

    def forward(self, x):
        x = x + self.token_mlp(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mlp(self.channel_norm(x))


class MLPMixer(nn.Module):
    def __init__(self, image_size=32, patch=8, dim=64, depth=2, classes=10):
        super().__init__()
        assert image_size % patch == 0
        tokens = (image_size // patch) ** 2
        self.patches = nn.Conv2d(3, dim, patch, stride=patch)
        self.blocks = nn.Sequential(*(MixerBlock(tokens, dim, 2 * tokens, 4 * dim) for _ in range(depth)))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, image):
        x = self.patches(image).flatten(2).transpose(1, 2)
        return self.head(self.norm(self.blocks(x)).mean(1))


if __name__ == "__main__":
    model = MLPMixer()
    y = model(torch.randn(2, 3, 32, 32))
    y.sum().backward()
    print("MLP-Mixer", tuple(y.shape))
