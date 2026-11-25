"""VGG-16：反复堆叠小的 3×3 卷积核。"""

import torch
from torch import nn


class VGG16(nn.Module):
    """VGG 配置 D（13 个卷积 + 3 个全连接权重层）。

    默认宽度/分类头对应原始大模型；CPU 练习可以缩小 `width_mult`、
    `hidden` 和 `pool_size`。输入通常是 [B, 3, 224, 224]。
    """

    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        hidden: int = 4096,
        pool_size: int = 7,
    ):
        super().__init__()
        if width_mult <= 0 or hidden <= 0 or pool_size <= 0:
            raise ValueError("width_mult、hidden 和 pool_size 必须为正")
        channels = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
                    512, 512, 512, "M", 512, 512, 512, "M"]
        layers = []
        in_channels = 3
        for item in channels:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels = max(1, round(item * width_mult))
                layers.extend([
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ])
                in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * pool_size * pool_size, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.avgpool(self.features(x)))


if __name__ == "__main__":
    model = VGG16(width_mult=0.125, hidden=64, pool_size=1, num_classes=10)
    print(model(torch.randn(2, 3, 64, 64)).shape)
