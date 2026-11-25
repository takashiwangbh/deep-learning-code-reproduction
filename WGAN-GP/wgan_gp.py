"""WGAN-GP：无 sigmoid 的 critic 和插值样本的梯度范数惩罚。"""

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, noise_dim=8, data_dim=16, hidden=32):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(noise_dim, hidden), nn.ReLU(), nn.Linear(hidden, data_dim))

    def forward(self, noise):
        return self.network(noise)


class Critic(nn.Module):
    def __init__(self, data_dim=16, hidden=32):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(data_dim, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1))

    def forward(self, data):
        return self.network(data)


def gradient_penalty(critic, real, fake):
    shape = [real.shape[0]] + [1] * (real.ndim - 1)
    alpha = torch.rand(shape, device=real.device, dtype=real.dtype)
    interpolated = (alpha * real.detach() + (1 - alpha) * fake.detach()).requires_grad_(True)
    scores = critic(interpolated)
    gradients = torch.autograd.grad(scores.sum(), interpolated, create_graph=True)[0]
    return ((gradients.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def critic_loss(critic, real, fake, penalty_weight=10.0):
    return critic(fake.detach()).mean() - critic(real).mean() + penalty_weight * gradient_penalty(critic, real, fake)


def generator_loss(critic, fake):
    return -critic(fake).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    generator, critic = Generator(), Critic()
    real, fake = torch.randn(4, 16), generator(torch.randn(4, 8))
    d_loss = critic_loss(critic, real, fake)
    g_loss = generator_loss(critic, fake)
    (d_loss + g_loss).backward()
    print("fake", tuple(fake.shape), "gradient penalty", round(gradient_penalty(critic, real, fake).item(), 3))
