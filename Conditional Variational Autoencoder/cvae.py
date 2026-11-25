"""同时包含条件后验 q(z|x,y) 和条件先验 p(z|y) 的 CVAE。"""

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalVAE(nn.Module):
    def __init__(self, input_dim=16, classes=3, latent_dim=4, hidden_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.embedding = nn.Embedding(classes, hidden_dim)
        self.posterior = nn.Sequential(nn.Linear(input_dim + hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2 * latent_dim))
        self.prior = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2 * latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim + hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    @staticmethod
    def split_normal(parameters):
        return parameters.chunk(2, dim=-1)  # 均值、log 方差

    def forward(self, x, y):
        c = self.embedding(y)
        q_mu, q_logvar = self.split_normal(self.posterior(torch.cat((x, c), dim=-1)))
        p_mu, p_logvar = self.split_normal(self.prior(c))
        z = q_mu + torch.exp(0.5 * q_logvar) * torch.randn_like(q_mu)
        logits = self.decoder(torch.cat((z, c), dim=-1))
        return logits, (q_mu, q_logvar), (p_mu, p_logvar)

    def loss(self, x, y):
        logits, (qm, qlv), (pm, plv) = self(x, y)
        reconstruction = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(-1)
        kl = 0.5 * (plv - qlv + (qlv.exp() + (qm - pm).square()) / plv.exp() - 1).sum(-1)
        return (reconstruction + kl).mean(), reconstruction.mean(), kl.mean()

    @torch.no_grad()
    def sample(self, y):
        c = self.embedding(y)
        mu, logvar = self.split_normal(self.prior(c))
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(torch.cat((z, c), dim=-1)).sigmoid()


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ConditionalVAE()
    x, y = torch.rand(4, 16), torch.tensor([0, 1, 2, 1])
    loss, reconstruction, kl = model.loss(x, y)
    loss.backward()
    print("generated", tuple(model.sample(y).shape), "reconstruction", round(reconstruction.item(), 3), "KL", round(kl.item(), 3))
