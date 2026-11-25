"""二值灰度 PixelCNN：A/B 掩码让每个像素只依赖此前像素。"""

import torch
from torch import nn
from torch.nn import functional as F


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type):
        assert mask_type in ("A", "B") and kernel_size % 2 == 1
        super().__init__(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        mask = torch.ones_like(self.weight)
        center = kernel_size // 2
        mask[:, :, center + 1 :, :] = 0
        mask[:, :, center, center + (mask_type == "B") :] = 0
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, padding=self.padding)


class PixelCNN(nn.Module):
    def __init__(self, hidden=24, blocks=3):
        super().__init__()
        self.input = MaskedConv2d(1, hidden, 7, "A")
        self.hidden = nn.Sequential(*(nn.Sequential(nn.ReLU(), MaskedConv2d(hidden, hidden, 3, "B")) for _ in range(blocks)))
        self.output = nn.Sequential(nn.ReLU(), nn.Conv2d(hidden, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, 1, 1))

    def forward(self, x):
        return self.output(self.hidden(F.relu(self.input(x))))

    def loss(self, binary_image):
        return F.binary_cross_entropy_with_logits(self(binary_image), binary_image)

    @torch.no_grad()
    def sample(self, batch=1, height=8, width=8):
        device = next(self.parameters()).device
        image = torch.zeros(batch, 1, height, width, device=device)
        for row in range(height):
            for col in range(width):
                probability = self(image)[:, 0, row, col].sigmoid()
                image[:, 0, row, col] = torch.bernoulli(probability)
        return image


if __name__ == "__main__":
    torch.manual_seed(0)
    model = PixelCNN()
    x = torch.randint(0, 2, (2, 1, 8, 8)).float()
    loss = model.loss(x)
    loss.backward()
    print("logits", tuple(model(x).shape), "sample", tuple(model.sample(1, 4, 4).shape), "BCE", round(loss.item(), 3))
