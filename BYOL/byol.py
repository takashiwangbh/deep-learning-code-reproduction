"""BYOL: online predictor learns a slowly moving target without negatives."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


def image_encoder(dim: int):
    return nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, dim))


def projector(dim: int):
    return nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                         nn.Linear(dim, dim))


class BYOL(nn.Module):
    def __init__(self, dim: int = 64, momentum: float = 0.99):
        super().__init__()
        self.online_encoder = image_encoder(dim)
        self.online_projector = projector(dim)
        self.predictor = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.target_encoder = deepcopy(self.online_encoder)
        self.target_projector = deepcopy(self.online_projector)
        self.momentum = momentum
        for module in (self.target_encoder, self.target_projector):
            for param in module.parameters():
                param.requires_grad_(False)

    @torch.no_grad()
    def update_target(self):
        """Call after the online optimizer step; target is an EMA, never SGD-updated."""
        for target, online in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            target.lerp_(online, 1 - self.momentum)
        for target, online in zip(self.target_projector.parameters(), self.online_projector.parameters()):
            target.lerp_(online, 1 - self.momentum)
        # BatchNorm running statistics are buffers rather than parameters.
        for target, online in zip(self.target_projector.buffers(), self.online_projector.buffers()):
            target.copy_(online)

    def forward(self, view_a: torch.Tensor, view_b: torch.Tensor):
        online_a = self.predictor(self.online_projector(self.online_encoder(view_a)))
        online_b = self.predictor(self.online_projector(self.online_encoder(view_b)))
        with torch.no_grad():
            self.target_projector.eval()  # target BN must not update from gradient batches
            target_a = self.target_projector(self.target_encoder(view_a))
            target_b = self.target_projector(self.target_encoder(view_b))
        loss_ab = 2 - 2 * F.cosine_similarity(online_a, target_b.detach(), dim=-1)
        loss_ba = 2 - 2 * F.cosine_similarity(online_b, target_a.detach(), dim=-1)
        return (loss_ab + loss_ba).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BYOL(dim=32)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    image = torch.randn(4, 3, 16, 16)
    loss = model(image, image + 0.05 * torch.randn_like(image))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.update_target()
    print("BYOL symmetric loss:", round(loss.item(), 4))
