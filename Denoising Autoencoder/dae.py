"""从带噪输入恢复干净输入的去噪自编码器。"""

import torch
from torch import nn
from torch.nn import functional as F


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def forward(self, corrupted):
        return self.decoder(self.encoder(corrupted))

    def loss(self, clean, noise_std=0.2):
        # 每次训练都重新抽噪声；目标始终是未污染的 clean。
        corrupted = clean + noise_std * torch.randn_like(clean)
        return F.mse_loss(self(corrupted), clean)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = DenoisingAutoencoder()
    x = torch.rand(4, 16)
    loss = model.loss(x)
    loss.backward()
    print("clean", tuple(x.shape), "reconstruction", tuple(model(x).shape), "loss", round(loss.item(), 4))
