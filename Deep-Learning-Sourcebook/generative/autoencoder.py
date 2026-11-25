"""最基本的自编码器：把输入压缩成瓶颈向量，再重建输入。

论文背景：Hinton & Salakhutdinov, Science 2006, DOI 10.1126/science.1127647。
本例展示核心结构；不包含原论文使用的逐层预训练。
"""

import torch
from torch import nn
from torch.nn import functional as F


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 256, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.flatten(start_dim=1)  # [B, C, H, W] -> [B, D]；向量输入也可用
        z = self.encoder(x)           # 瓶颈：强制网络保留有用信息
        reconstruction = self.decoder(z)
        return reconstruction, z


def reconstruction_loss(reconstruction: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """连续值输入的均方误差。输入若为二值图像，也可改用 BCE。"""
    return F.mse_loss(reconstruction, x.flatten(start_dim=1))


if __name__ == "__main__":
    model = Autoencoder()
    images = torch.rand(4, 1, 28, 28)
    output, code = model(images)
    loss = reconstruction_loss(output, images)
    loss.backward()
    print(f"reconstruction={tuple(output.shape)}, latent={tuple(code.shape)}, loss={loss.item():.4f}")
