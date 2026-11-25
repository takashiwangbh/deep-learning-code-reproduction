"""4× SRGAN 教学版：残差超分生成器、判别器和内容/对抗目标。"""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return x + self.norm2(self.conv2(F.relu(self.norm1(self.conv1(x)))))


class SRGenerator(nn.Module):
    def __init__(self, channels=3, width=16, blocks=2):
        super().__init__()
        self.first = nn.Conv2d(channels, width, 9, padding=4)
        self.first_activation = nn.PReLU()
        self.residuals = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.trunk = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width))
        self.upsample = nn.Sequential(
            nn.Conv2d(width, width * 4, 3, padding=1), nn.PixelShuffle(2), nn.PReLU(),
            nn.Conv2d(width, width * 4, 3, padding=1), nn.PixelShuffle(2), nn.PReLU(),
            nn.Conv2d(width, channels, 9, padding=4), nn.Tanh(),
        )

    def forward(self, low_resolution):
        first = self.first_activation(self.first(low_resolution))
        return self.upsample(first + self.trunk(self.residuals(first)))


class SRDiscriminator(nn.Module):
    def __init__(self, channels=3, width=16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(width, width * 2, 4, 2, 1), nn.BatchNorm2d(width * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(width * 2, width * 4, 4, 2, 1), nn.BatchNorm2d(width * 4), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 4, 1),
        )

    def forward(self, high_resolution):
        return self.network(high_resolution)


class FeatureExtractor(nn.Module):
    """本地可读的 VGG 风格特征架构；实际感知训练须载入有意义的预训练权重。"""

    def __init__(self, channels=3):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, image):
        return self.network(image)


def losses(generator, discriminator, feature_extractor, low_resolution, high_resolution, adversarial_weight=1e-3):
    fake = generator(low_resolution)
    d_loss = F.softplus(-discriminator(high_resolution)).mean() + F.softplus(discriminator(fake.detach())).mean()
    content = F.mse_loss(feature_extractor(fake), feature_extractor(high_resolution).detach())
    g_loss = content + adversarial_weight * F.softplus(-discriminator(fake)).mean()
    return d_loss, g_loss, content


if __name__ == "__main__":
    torch.manual_seed(0)
    g, d, f = SRGenerator(), SRDiscriminator(), FeatureExtractor()
    low, high = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 32, 32)
    d_loss, g_loss, content = losses(g, d, f, low, high)
    (d_loss + g_loss).backward()
    print("4x super-resolution", tuple(g(low).shape), "content", round(content.item(), 4))
