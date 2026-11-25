"""GoogLeNet (Inception v1)：并行多尺度卷积与 1×1 通道降维。"""

import torch
from torch import nn


def _conv(in_channels: int, out_channels: int, kernel: int, **kwargs) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, **kwargs),
        nn.ReLU(inplace=True),
    )


class Inception(nn.Module):
    """四路并行：1×1、1×1→3×3、1×1→5×5、池化→1×1。

    1×1 降维把昂贵的 3×3/5×5 计算限制在较少通道上。
    """

    def __init__(self, in_channels: int, widths: tuple[int, int, int, int, int, int]):
        super().__init__()
        c1, c3red, c3, c5red, c5, pool = widths
        self.branch1 = _conv(in_channels, c1, 1)
        self.branch3 = nn.Sequential(
            _conv(in_channels, c3red, 1), _conv(c3red, c3, 3, padding=1)
        )
        self.branch5 = nn.Sequential(
            _conv(in_channels, c5red, 1), _conv(c5red, c5, 5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1), _conv(in_channels, pool, 1)
        )
        self.out_channels = c1 + c3 + c5 + pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.branch1(x), self.branch3(x), self.branch5(x), self.branch_pool(x)],
            dim=1,
        )


class AuxiliaryClassifier(nn.Module):
    """训练时的辅助分类头，给中段网络一条更短的梯度路径。"""

    def __init__(self, in_channels: int, num_classes: int, width_mult: float):
        super().__init__()
        reduced = max(1, round(128 * width_mult))
        hidden = max(16, round(1024 * width_mult))
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            _conv(in_channels, reduced, 1),
            nn.Flatten(),
            nn.Linear(reduced * 4 * 4, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GoogLeNet(nn.Module):
    """九个 Inception 模块。训练且 aux_logits=True 时返回主、辅1、辅2 logits。"""

    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0,
                 aux_logits: bool = True):
        super().__init__()
        if width_mult <= 0:
            raise ValueError("width_mult 必须为正")
        self.aux_logits = aux_logits

        def c(value: int) -> int:
            return max(1, round(value * width_mult))

        self.stem = nn.Sequential(
            _conv(3, c(64), 7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0),
            _conv(c(64), c(64), 1),
            _conv(c(64), c(192), 3, padding=1),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        specs = [
            (64, 96, 128, 16, 32, 32),   # 3a
            (128, 128, 192, 32, 96, 64), # 3b
            (192, 96, 208, 16, 48, 64),  # 4a
            (160, 112, 224, 24, 64, 64), # 4b
            (128, 128, 256, 24, 64, 64), # 4c
            (112, 144, 288, 32, 64, 64), # 4d
            (256, 160, 320, 32, 128, 128), # 4e
            (256, 160, 320, 32, 128, 128), # 5a
            (384, 192, 384, 48, 128, 128), # 5b
        ]
        modules = []
        in_channels = c(192)
        aux_channels = []
        for index, spec in enumerate(specs):
            module = Inception(in_channels, tuple(c(v) for v in spec))
            modules.append(module)
            in_channels = module.out_channels
            if index in (2, 5):
                aux_channels.append(in_channels)
        self.inceptions = nn.ModuleList(modules)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        if aux_logits:
            self.aux1 = AuxiliaryClassifier(aux_channels[0], num_classes, width_mult)
            self.aux2 = AuxiliaryClassifier(aux_channels[1], num_classes, width_mult)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        aux1 = aux2 = None
        for index, inception in enumerate(self.inceptions):
            if index in (2, 7):
                x = self.pool(x)
            x = inception(x)
            if self.training and self.aux_logits:
                if index == 2:
                    aux1 = self.aux1(x)
                elif index == 5:
                    aux2 = self.aux2(x)
        logits = self.head(x)
        return (logits, aux1, aux2) if self.training and self.aux_logits else logits


if __name__ == "__main__":
    model = GoogLeNet(num_classes=10, width_mult=0.125, aux_logits=False)
    print(model(torch.randn(2, 3, 64, 64)).shape)
