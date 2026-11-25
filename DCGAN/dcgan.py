"""DCGAN 生成器、判别器和非饱和 GAN 训练目标。

Radford et al., Unsupervised Representation Learning with DCGANs:
https://arxiv.org/abs/1511.06434
"""

import torch
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    """z: [B, latent_dim, 1, 1] -> RGB: [B, 3, 64, 64]，像素范围 [-1, 1]。"""

    def __init__(self, latent_dim: int = 100, base_channels: int = 64, image_channels: int = 3):
        super().__init__()
        c = base_channels
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, c * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(c * 8), nn.ReLU(True),                  # 4 x 4
            nn.ConvTranspose2d(c * 8, c * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 4), nn.ReLU(True),                  # 8 x 8
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 2), nn.ReLU(True),                  # 16 x 16
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(True),                      # 32 x 32
            nn.ConvTranspose2d(c, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),                                             # 64 x 64
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            z = z[:, :, None, None]
        return self.net(z)


class Discriminator(nn.Module):
    """返回未经过 sigmoid 的真假 logits，避免 BCE 中的数值不稳定。"""

    def __init__(self, base_channels: int = 64, image_channels: int = 3):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, c, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),                       # 32 x 32
            nn.Conv2d(c, c * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 2), nn.LeakyReLU(0.2, inplace=True), # 16 x 16
            nn.Conv2d(c * 2, c * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 4), nn.LeakyReLU(0.2, inplace=True), # 8 x 8
            nn.Conv2d(c * 4, c * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 8), nn.LeakyReLU(0.2, inplace=True), # 4 x 4
            nn.Conv2d(c * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images).flatten(start_dim=1).squeeze(1)


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))


def generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    # 非饱和目标 -log D(G(z))，使生成器在早期也能收到有效梯度。
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


if __name__ == "__main__":
    generator, discriminator = Generator(base_channels=8), Discriminator(base_channels=8)
    fake = generator(torch.randn(2, 100))
    d_loss = discriminator_loss(discriminator(torch.randn_like(fake)), discriminator(fake.detach()))
    d_loss.backward()
    g_loss = generator_loss(discriminator(fake))
    g_loss.backward()
    print(f"fake={tuple(fake.shape)}, D={d_loss.item():.3f}, G={g_loss.item():.3f}")
