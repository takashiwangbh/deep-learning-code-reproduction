"""ControlNet mechanism: locked denoiser, trainable branch and zero convolutions."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class TinyDenoiser(nn.Module):
    def __init__(self, image_channels: int = 3, hidden: int = 32):
        super().__init__()
        self.stem = nn.Conv2d(image_channels, hidden, 3, padding=1)
        self.time = nn.Linear(1, hidden)
        self.block = nn.Sequential(nn.SiLU(), nn.Conv2d(hidden, hidden, 3, padding=1),
                                   nn.SiLU(), nn.Conv2d(hidden, hidden, 3, padding=1))
        self.head = nn.Conv2d(hidden, image_channels, 3, padding=1)

    def features(self, noisy: torch.Tensor, time: torch.Tensor):
        hidden = self.stem(noisy) + self.time(time.float()[:, None])[:, :, None, None]
        return self.block(hidden)

    def forward(self, noisy: torch.Tensor, time: torch.Tensor):
        return self.head(self.features(noisy, time))


class ControlNet(nn.Module):
    def __init__(self, pretrained_backbone: TinyDenoiser, control_channels: int = 1):
        super().__init__()
        self.locked = deepcopy(pretrained_backbone)
        for parameter in self.locked.parameters():
            parameter.requires_grad_(False)
        hidden = self.locked.stem.out_channels
        self.trainable_stem = deepcopy(pretrained_backbone.stem)
        self.trainable_time = deepcopy(pretrained_backbone.time)
        self.trainable_block = deepcopy(pretrained_backbone.block)
        self.condition_encoder = nn.Sequential(nn.Conv2d(control_channels, hidden, 3, padding=1),
                                               nn.SiLU())
        self.zero_condition = nn.Conv2d(hidden, hidden, 1)
        self.zero_residual = nn.Conv2d(hidden, hidden, 1)
        for layer in (self.zero_condition, self.zero_residual):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, noisy: torch.Tensor, time: torch.Tensor, control: torch.Tensor):
        """[B,C,H,W] noisy and [B,K,H,W] control -> predicted noise [B,C,H,W]."""
        if noisy.shape[0] != control.shape[0] or noisy.shape[-2:] != control.shape[-2:]:
            raise ValueError("Control map must match image batch and spatial size")
        base_features = self.locked.features(noisy, time)
        branch = self.trainable_stem(noisy)
        branch = branch + self.trainable_time(time.float()[:, None])[:, :, None, None]
        branch = branch + self.zero_condition(self.condition_encoder(control))
        residual = self.zero_residual(self.trainable_block(branch))
        return self.locked.head(base_features + residual)


if __name__ == "__main__":
    torch.manual_seed(0)
    base = TinyDenoiser()
    model = ControlNet(base)
    noisy = torch.randn(2, 3, 8, 8)
    time = torch.rand(2)
    control = torch.randn(2, 1, 8, 8)
    error = (model(noisy, time, control) - torch.randn_like(noisy)).square().mean()
    error.backward()
    print("ControlNet output/frozen gradient:", tuple(model(noisy, time, control).shape),
          all(parameter.grad is None for parameter in model.locked.parameters()))
