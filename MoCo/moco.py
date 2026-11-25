"""MoCo v1: query encoder, momentum key encoder and a FIFO negative queue."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class TinyImageEncoder(nn.Module):
    def __init__(self, channels: int = 3, dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, dim))

    def forward(self, images: torch.Tensor):
        return F.normalize(self.net(images), dim=-1)


class MoCo(nn.Module):
    def __init__(self, dim: int = 32, queue_size: int = 64, momentum: float = 0.999,
                 temperature: float = 0.07):
        super().__init__()
        self.query_encoder = TinyImageEncoder(dim=dim)
        self.key_encoder = deepcopy(self.query_encoder)
        for param in self.key_encoder.parameters():
            param.requires_grad_(False)
        self.momentum, self.temperature = momentum, temperature
        self.register_buffer("queue", F.normalize(torch.randn(queue_size, dim), dim=-1))
        self.register_buffer("queue_pointer", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update_key_encoder(self):
        for key, query in zip(self.key_encoder.parameters(), self.query_encoder.parameters()):
            key.lerp_(query, 1 - self.momentum)

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor):
        # Works even when the current batch wraps around the circular queue.
        size = self.queue.shape[0]
        for key in keys.detach():
            index = int(self.queue_pointer)
            self.queue[index].copy_(key)
            self.queue_pointer.fill_((index + 1) % size)

    def forward(self, view_q: torch.Tensor, view_k: torch.Tensor):
        """Two augmentations [B,C,H,W] -> InfoNCE scalar loss."""
        query = self.query_encoder(view_q)
        with torch.no_grad():
            self.update_key_encoder()
            key = self.key_encoder(view_k)
        positive = (query * key).sum(dim=-1, keepdim=True)
        negative = query @ self.queue.detach().clone().T
        logits = torch.cat((positive, negative), dim=-1) / self.temperature
        labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        loss = F.cross_entropy(logits, labels)
        self.enqueue(key)
        return loss


if __name__ == "__main__":
    torch.manual_seed(0)
    model = MoCo(queue_size=12)
    images = torch.randn(4, 3, 16, 16)
    loss = model(images, images + 0.05 * torch.randn_like(images))
    loss.backward()
    print("MoCo contrastive loss / queue pointer:", round(loss.item(), 4), int(model.queue_pointer))
