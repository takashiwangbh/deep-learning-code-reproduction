"""原始全连接 GAN：生成器、判别器和交替优化所需的两个损失。"""

import torch
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, noise_dim=8, data_dim=16, hidden=32):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(noise_dim, hidden), nn.ReLU(), nn.Linear(hidden, data_dim), nn.Tanh())

    def forward(self, noise):
        return self.network(noise)


class Discriminator(nn.Module):
    def __init__(self, data_dim=16, hidden=32):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(data_dim, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1))

    def forward(self, data):
        return self.network(data)


def discriminator_loss(discriminator, real, fake):
    real_logits = discriminator(real)
    fake_logits = discriminator(fake.detach())
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()


def generator_loss(discriminator, fake):
    # 非饱和式 -log D(G(z)) 与原始 minimax 有相同理想平衡点，梯度更强。
    return F.softplus(-discriminator(fake)).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    generator, discriminator = Generator(), Discriminator()
    real, noise = torch.rand(4, 16) * 2 - 1, torch.randn(4, 8)
    fake = generator(noise)
    d_loss = discriminator_loss(discriminator, real, fake)
    g_loss = generator_loss(discriminator, fake)
    (d_loss + g_loss).backward()
    print("fake", tuple(fake.shape), "D loss", round(d_loss.item(), 3), "G loss", round(g_loss.item(), 3))
