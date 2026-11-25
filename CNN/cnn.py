"""Small CNN that exposes convolution, activation, pooling, and classifier."""

import torch
from torch import nn


class BasicCNN(nn.Module):
    def __init__(self, classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


if __name__ == "__main__":
    model = BasicCNN()
    y = model(torch.randn(2, 1, 28, 28))
    y.sum().backward()
    print("CNN logits:", tuple(y.shape))
