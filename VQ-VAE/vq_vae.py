"""最近邻离散码本 + straight-through 梯度的向量量化自编码器。"""

import torch
from torch import nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, codes=16, dim=8, commitment=0.25):
        super().__init__()
        self.codebook = nn.Embedding(codes, dim)
        self.commitment = commitment

    def forward(self, z_e):
        batch, dim, height, width = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, dim)
        embeddings = self.codebook.weight
        distances = flat.square().sum(-1, keepdim=True) - 2 * flat @ embeddings.T + embeddings.square().sum(-1)
        indices = distances.argmin(-1)
        z_q = self.codebook(indices).reshape(batch, height, width, dim).permute(0, 3, 1, 2).contiguous()
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        z_st = z_e + (z_q - z_e).detach()  # 前向是 z_q，反向对编码器是恒等梯度。
        return z_st, indices.reshape(batch, height, width), codebook_loss + self.commitment * commitment_loss


class VQVAE(nn.Module):
    def __init__(self, channels=1, width=16, code_dim=8, codes=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(channels, width, 4, 2, 1), nn.ReLU(), nn.Conv2d(width, code_dim, 4, 2, 1))
        self.quantizer = VectorQuantizer(codes, code_dim)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(code_dim, width, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(width, channels, 4, 2, 1))

    def forward(self, x):
        z_q, indices, quantizer_loss = self.quantizer(self.encoder(x))
        return self.decoder(z_q), indices, quantizer_loss

    def loss(self, x):
        reconstructed, indices, quantizer_loss = self(x)
        return F.mse_loss(reconstructed, x) + quantizer_loss, indices


if __name__ == "__main__":
    torch.manual_seed(0)
    model = VQVAE()
    x = torch.rand(2, 1, 16, 16)
    loss, indices = model.loss(x)
    loss.backward()
    print("image", tuple(model(x)[0].shape), "code indices", tuple(indices.shape), "loss", round(loss.item(), 3))
