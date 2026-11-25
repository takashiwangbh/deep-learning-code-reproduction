"""变分自编码器：重参数化采样 + 重建项 + KL 正则项。

Kingma & Welling, Auto-Encoding Variational Bayes: https://arxiv.org/abs/1312.6114
"""

import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 400, latent_dim: int = 20):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),  # 输出 logits，不预先 sigmoid
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # z = μ + σ ε, ε ~ N(0, I)。梯度经 μ 和 σ 传回编码器。
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x.flatten(start_dim=1))
        mu, logvar = self.to_mu(hidden), self.to_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def sample(self, count: int, device: torch.device | str | None = None) -> torch.Tensor:
        if device is None:
            device = self.to_mu.weight.device
        z = torch.randn(count, self.to_mu.out_features, device=device)
        return self.decoder(z).sigmoid()


def elbo_loss(
    logits: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回每个样本平均的负 ELBO、重建损失、KL 散度。

    假定像素位于 [0, 1]，使用 Bernoulli 似然。loss = -E[log p(x|z)] + KL(q(z|x)||N(0,I))。
    """
    target = target.flatten(start_dim=1)
    reconstruction = F.binary_cross_entropy_with_logits(logits, target, reduction="sum") / target.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp()) / target.size(0)
    return reconstruction + kl, reconstruction, kl


if __name__ == "__main__":
    model = VAE()
    x = torch.rand(4, 1, 28, 28)
    logits, mu, logvar = model(x)
    loss, reconstruction, kl = elbo_loss(logits, x, mu, logvar)
    loss.backward()
    print(f"loss={loss.item():.3f}, reconstruction={reconstruction.item():.3f}, KL={kl.item():.3f}")
