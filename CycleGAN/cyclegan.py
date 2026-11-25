"""无配对图像翻译：双向生成器、两个 Patch 判别器和循环一致性。"""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.InstanceNorm2d(channels), nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1), nn.InstanceNorm2d(channels))

    def forward(self, x):
        return x + self.network(x)


class ResnetGenerator(nn.Module):
    def __init__(self, channels=3, width=16, blocks=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, width, 7, padding=3), nn.InstanceNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, width * 2, 4, 2, 1), nn.InstanceNorm2d(width * 2), nn.ReLU(),
            *(ResidualBlock(width * 2) for _ in range(blocks)),
            nn.ConvTranspose2d(width * 2, width, 4, 2, 1), nn.InstanceNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, channels, 7, padding=3), nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, channels=3, width=16):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(channels, width, 4, 2, 1), nn.LeakyReLU(0.2), nn.Conv2d(width, width * 2, 4, 2, 1), nn.InstanceNorm2d(width * 2), nn.LeakyReLU(0.2), nn.Conv2d(width * 2, 1, 3, padding=1))

    def forward(self, x):
        return self.network(x)


def losses(g_ab, g_ba, d_a, d_b, real_a, real_b, cycle_weight=10.0, identity_weight=5.0):
    fake_b, fake_a = g_ab(real_a), g_ba(real_b)
    cycle_a, cycle_b = g_ba(fake_b), g_ab(fake_a)
    d_loss = (F.softplus(-d_a(real_a)).mean() + F.softplus(d_a(fake_a.detach())).mean()
              + F.softplus(-d_b(real_b)).mean() + F.softplus(d_b(fake_b.detach())).mean())
    adversarial = F.softplus(-d_a(fake_a)).mean() + F.softplus(-d_b(fake_b)).mean()
    cycle = F.l1_loss(cycle_a, real_a) + F.l1_loss(cycle_b, real_b)
    identity = F.l1_loss(g_ba(real_a), real_a) + F.l1_loss(g_ab(real_b), real_b)
    return d_loss, adversarial + cycle_weight * cycle + identity_weight * identity, cycle


if __name__ == "__main__":
    torch.manual_seed(0)
    g_ab, g_ba, d_a, d_b = ResnetGenerator(), ResnetGenerator(), PatchDiscriminator(), PatchDiscriminator()
    a, b = torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16)
    d_loss, g_loss, cycle = losses(g_ab, g_ba, d_a, d_b, a, b)
    (d_loss + g_loss).backward()
    print("A to B", tuple(g_ab(a).shape), "cycle L1", round(cycle.item(), 3))
