"""条件 GAN：标签同时进入生成器和判别器。"""

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim=8, classes=3, data_dim=16, hidden=32):
        super().__init__()
        self.classes = classes
        self.network = nn.Sequential(nn.Linear(noise_dim + classes, hidden), nn.ReLU(), nn.Linear(hidden, data_dim), nn.Tanh())

    def forward(self, noise, labels):
        condition = F.one_hot(labels, self.classes).to(noise.dtype)
        return self.network(torch.cat((noise, condition), dim=-1))


class ConditionalDiscriminator(nn.Module):
    def __init__(self, classes=3, data_dim=16, hidden=32):
        super().__init__()
        self.classes = classes
        self.network = nn.Sequential(nn.Linear(data_dim + classes, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1))

    def forward(self, data, labels):
        condition = F.one_hot(labels, self.classes).to(data.dtype)
        return self.network(torch.cat((data, condition), dim=-1))


def losses(generator, discriminator, real, labels, noise):
    fake = generator(noise, labels)
    d_real = discriminator(real, labels)
    d_fake = discriminator(fake.detach(), labels)
    d_loss = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
    g_loss = F.softplus(-discriminator(fake, labels)).mean()
    return d_loss, g_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    g, d = ConditionalGenerator(), ConditionalDiscriminator()
    real, labels, noise = torch.rand(4, 16) * 2 - 1, torch.tensor([0, 1, 2, 1]), torch.randn(4, 8)
    d_loss, g_loss = losses(g, d, real, labels, noise)
    (d_loss + g_loss).backward()
    print("conditional fake", tuple(g(noise, labels).shape), "D loss", round(d_loss.item(), 3), "G loss", round(g_loss.item(), 3))
