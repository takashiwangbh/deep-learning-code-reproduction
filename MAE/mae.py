"""Masked Autoencoder: encode only visible patches, decode all, score masked pixels."""

import torch
from torch import nn
from torch.nn import functional as F


class MaskedAutoencoder(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 8, channels: int = 3,
                 dim: int = 64, mask_ratio: float = 0.75):
        super().__init__()
        if image_size % patch_size or not 0 < mask_ratio < 1:
            raise ValueError("Image size must divide into patches and mask ratio must be in (0,1)")
        self.patch_size, self.channels, self.mask_ratio = patch_size, channels, mask_ratio
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size * patch_size
        self.patch_embed = nn.Linear(self.patch_dim, dim)
        self.encoder_position = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        self.decoder_position = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        encoder_block = nn.TransformerEncoderLayer(dim, 4, dim * 2, dropout=0, batch_first=True)
        decoder_block = nn.TransformerEncoderLayer(dim, 4, dim * 2, dropout=0, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_block, 2)
        self.decoder = nn.TransformerEncoder(decoder_block, 1)
        self.to_pixels = nn.Linear(dim, self.patch_dim)

    def patchify(self, images: torch.Tensor):
        if images.shape[1] != self.channels:
            raise ValueError("Unexpected channel count")
        patches = F.unfold(images, kernel_size=self.patch_size, stride=self.patch_size)
        if patches.shape[-1] != self.num_patches:
            raise ValueError("Unexpected image size")
        return patches.transpose(1, 2)

    def forward(self, images: torch.Tensor):
        target = self.patchify(images)
        batch, count, _ = target.shape
        visible_count = max(1, int(count * (1 - self.mask_ratio)))
        order = torch.rand(batch, count, device=images.device).argsort(dim=-1)
        visible_ids = order[:, :visible_count]
        visible = (self.patch_embed(target) + self.encoder_position).gather(
            1, visible_ids[..., None].expand(-1, -1, self.encoder_position.shape[-1]))
        encoded = self.encoder(visible)
        restored = self.mask_token.expand(batch, count, -1).clone()
        restored.scatter_(1, visible_ids[..., None].expand_as(encoded), encoded)
        predicted = self.to_pixels(self.decoder(restored + self.decoder_position))
        masked = torch.ones(batch, count, device=images.device, dtype=torch.bool)
        masked.scatter_(1, visible_ids, False)
        # Paper's normalized-pixel target, with epsilon for constant patches.
        normalized = (target - target.mean(dim=-1, keepdim=True)) / torch.sqrt(
            target.var(dim=-1, keepdim=True, unbiased=False) + 1e-6)
        loss = ((predicted - normalized) ** 2).mean(dim=-1)[masked].mean()
        return predicted, masked, loss


if __name__ == "__main__":
    torch.manual_seed(0)
    model = MaskedAutoencoder()
    prediction, mask, loss = model(torch.randn(2, 3, 32, 32))
    loss.backward()
    print("MAE patch predictions/masked count:", tuple(prediction.shape), int(mask.sum()))
