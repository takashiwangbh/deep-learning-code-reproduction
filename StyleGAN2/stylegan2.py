"""小尺寸 StyleGAN2：映射网络、调制/解调卷积、噪声注入及 R1 正则。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, kernel=3, demodulate=True, upsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.demodulate = demodulate
        self.upsample = upsample
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel, kernel) / math.sqrt(in_channels * kernel * kernel))
        self.affine = nn.Linear(style_dim, in_channels)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x, style):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        batch, _, height, width = x.shape
        scales = 1 + self.affine(style)
        weights = self.weight[None] * scales[:, None, :, None, None]
        if self.demodulate:
            normalizer = torch.rsqrt(weights.square().sum((2, 3, 4)) + 1e-8)
            weights = weights * normalizer[:, :, None, None, None]
        grouped_weights = weights.reshape(batch * self.out_channels, self.in_channels, self.kernel, self.kernel)
        x = F.conv2d(x.reshape(1, batch * self.in_channels, height, width), grouped_weights, padding=self.kernel // 2, groups=batch)
        return x.reshape(batch, self.out_channels, height, width)


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.strength = nn.Parameter(torch.zeros(()))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.shape[0], 1, *x.shape[-2:], device=x.device, dtype=x.dtype)
        return x + self.strength * noise


class StyleGenerator(nn.Module):
    def __init__(self, latent_dim=16, style_dim=16, width=16, channels=3):
        super().__init__()
        self.mapping = nn.Sequential(nn.Linear(latent_dim, style_dim), nn.LeakyReLU(0.2), nn.Linear(style_dim, style_dim))
        self.constant = nn.Parameter(torch.randn(1, width, 4, 4))
        self.conv1 = ModulatedConv2d(width, width, style_dim)
        self.conv2 = ModulatedConv2d(width, width, style_dim, upsample=True)
        self.noise1, self.noise2 = NoiseInjection(), NoiseInjection()
        self.to_rgb = ModulatedConv2d(width, channels, style_dim, kernel=1, demodulate=False)

    def synthesis(self, style, noise1=None, noise2=None):
        x = self.constant.expand(style.shape[0], -1, -1, -1)
        x = F.leaky_relu(self.noise1(self.conv1(x, style), noise1), 0.2)
        x = F.leaky_relu(self.noise2(self.conv2(x, style), noise2), 0.2)
        return self.to_rgb(x, style).tanh()

    def forward(self, latent):
        return self.synthesis(self.mapping(latent))


class StyleDiscriminator(nn.Module):
    def __init__(self, channels=3, width=16):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(channels, width, 3, padding=1), nn.LeakyReLU(0.2), nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.LeakyReLU(0.2), nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.LeakyReLU(0.2), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 2, 1))

    def forward(self, image):
        return self.network(image)


def r1_penalty(discriminator, real):
    real = real.detach().requires_grad_(True)
    gradient = torch.autograd.grad(discriminator(real).sum(), real, create_graph=True)[0]
    return gradient.square().flatten(1).sum(1).mean()


def path_length_penalty(generator, latent, target_length=0.0):
    """教学版固定目标长度；实际训练用移动平均更新 target_length。"""
    style = generator.mapping(latent)
    image = generator.synthesis(style)
    noise = torch.randn_like(image) / math.sqrt(image.shape[2] * image.shape[3])
    path = torch.autograd.grad((image * noise).sum(), style, create_graph=True)[0]
    lengths = path.square().sum(-1).sqrt()
    return (lengths - target_length).square().mean(), lengths.detach()


def adversarial_losses(generator, discriminator, real, latent, r1_weight=1.0):
    fake = generator(latent)
    d_loss = F.softplus(-discriminator(real)).mean() + F.softplus(discriminator(fake.detach())).mean()
    d_loss = d_loss + 0.5 * r1_weight * r1_penalty(discriminator, real)
    g_loss = F.softplus(-discriminator(fake)).mean()
    return d_loss, g_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    g, d = StyleGenerator(), StyleDiscriminator()
    z, real = torch.randn(2, 16), torch.randn(2, 3, 8, 8)
    d_loss, g_loss = adversarial_losses(g, d, real, z)
    path_penalty, _ = path_length_penalty(g, z)
    (d_loss + g_loss + path_penalty).backward()
    print("styled image", tuple(g(z).shape), "R1", round(r1_penalty(d, real).item(), 4))
