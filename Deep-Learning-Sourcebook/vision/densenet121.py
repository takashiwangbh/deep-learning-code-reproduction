"""DenseNet-121：层层拼接，显式复用此前所有特征图。"""

import torch
from torch import nn


class DenseLayer(nn.Module):
    """瓶颈 1×1 卷积 + 3×3 卷积；只产生 `growth_rate` 个新通道。"""

    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4):
        super().__init__()
        intermediate = bn_size * growth_rate
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, intermediate, 1, bias=False),
            nn.BatchNorm2d(intermediate),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate, growth_rate, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, growth_rate: int):
        super().__init__()
        self.layers = nn.ModuleList(
            DenseLayer(in_channels + i * growth_rate, growth_rate)
            for i in range(num_layers)
        )
        self.out_channels = in_channels + num_layers * growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            new_features = layer(x)
            # 核心公式：x_l = H_l([x_0, x_1, ..., x_{l-1}])。
            x = torch.cat((x, new_features), dim=1)
        return x


class Transition(nn.Sequential):
    """压缩通道并将空间宽高减半。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2),
        )


class DenseNet121(nn.Module):
    """默认 block=(6,12,24,16)、增长率 32，即 DenseNet-121。"""

    def __init__(
        self,
        num_classes: int = 1000,
        growth_rate: int = 32,
        block_config: tuple[int, ...] = (6, 12, 24, 16),
        num_init_features: int = 64,
        compression: float = 0.5,
    ):
        super().__init__()
        if growth_rate <= 0 or num_init_features <= 0 or not 0 < compression <= 1:
            raise ValueError("增长率、初始通道和压缩率无效")
        self.stem = nn.Sequential(
            nn.Conv2d(3, num_init_features, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        channels = num_init_features
        blocks = []
        for index, num_layers in enumerate(block_config):
            block = DenseBlock(channels, num_layers, growth_rate)
            blocks.append(block)
            channels = block.out_channels
            if index < len(block_config) - 1:
                compressed = max(1, int(channels * compression))
                blocks.append(Transition(channels, compressed))
                channels = compressed
        self.features = nn.Sequential(*blocks, nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(self.stem(x))).flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    model = DenseNet121(num_classes=10, growth_rate=4,
                        block_config=(1, 1, 1, 1), num_init_features=8)
    print(model(torch.randn(2, 3, 64, 64)).shape)
