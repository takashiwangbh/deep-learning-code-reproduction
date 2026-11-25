"""ResNet-50 bottleneck network with the original 3-4-6-3 stage layout."""

import torch
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        out_channels = channels * self.expansion
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (nn.Identity() if stride == 1 and in_channels == out_channels else
                         nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_channels)))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.activation(self.main(x) + self.shortcut(x))


class ResNet50(nn.Module):
    def __init__(self, classes: int = 1000, base_width: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        current = base_width
        stages = []
        for stage_index, (channels, count) in enumerate(zip(
                [base_width, base_width * 2, base_width * 4, base_width * 8], [3, 4, 6, 3])):
            blocks = []
            for block_index in range(count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(Bottleneck(current, channels, stride))
                current = channels * Bottleneck.expansion
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current, classes)

    def forward(self, image: torch.Tensor):
        feature = self.stem(image)
        for stage in self.stages:
            feature = stage(feature)
        return self.classifier(self.pool(feature).flatten(1))


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ResNet50(classes=7, base_width=8)
    logits = model(torch.randn(2, 3, 64, 64))
    logits.sum().backward()
    print("ResNet-50 bottleneck counts:", [len(stage) for stage in model.stages],
          "logits:", tuple(logits.shape))
