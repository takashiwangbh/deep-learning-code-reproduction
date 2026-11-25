"""成对图像翻译：小型 U-Net、PatchGAN 与 adversarial + L1 损失。"""

import torch
from torch import nn
from torch.nn import functional as F


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, width=16):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, width, 4, 2, 1), nn.LeakyReLU(0.2))
        self.down2 = nn.Sequential(nn.Conv2d(width, width * 2, 4, 2, 1), nn.InstanceNorm2d(width * 2), nn.LeakyReLU(0.2))
        self.middle = nn.Sequential(nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.ReLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(width * 2, width, 4, 2, 1), nn.InstanceNorm2d(width), nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(width * 2, width, 4, 2, 1), nn.ReLU())
        self.out = nn.Sequential(nn.Conv2d(width, out_channels, 3, padding=1), nn.Tanh())

    def forward(self, source):
        x1 = self.down1(source)
        x2 = self.down2(x1)
        u1 = self.up1(self.middle(x2))
        return self.out(self.up2(torch.cat((u1, x1), dim=1)))


class PatchDiscriminator(nn.Module):
    def __init__(self, source_channels=3, target_channels=3, width=16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(source_channels + target_channels, width, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(width, width * 2, 4, 2, 1), nn.InstanceNorm2d(width * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(width * 2, 1, 3, padding=1),
        )

    def forward(self, source, target):
        return self.network(torch.cat((source, target), dim=1))


def losses(generator, discriminator, source, paired_target, l1_weight=100.0):
    fake = generator(source)
    real_logits = discriminator(source, paired_target)
    fake_logits = discriminator(source, fake.detach())
    d_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
    g_loss = F.softplus(-discriminator(source, fake)).mean() + l1_weight * F.l1_loss(fake, paired_target)
    return d_loss, g_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    g, d = UNetGenerator(), PatchDiscriminator()
    source, target = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32).tanh()
    d_loss, g_loss = losses(g, d, source, target)
    (d_loss + g_loss).backward()
    print("translated", tuple(g(source).shape), "patch scores", tuple(d(source, target).shape))
