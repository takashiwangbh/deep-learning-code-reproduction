"""MobileNetV2：倒残差块、深度可分离卷积和线性瓶颈。"""

import torch
from torch import nn


def _make_divisible(value: float, divisor: int = 8) -> int:
    """令通道数适合常见硬件向量宽度，同时避免过度向下取整。"""
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    return rounded + divisor if rounded < 0.9 * value else rounded


class InvertedResidual(nn.Module):
    """窄输入 → 1×1 扩张 → 深度卷积 → 线性 1×1 压缩。

    只有步长为 1 且输入/输出通道相同时才能逐元素相加。
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int, expansion: int):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride 只能是 1 或 2")
        hidden = int(round(in_channels * expansion))
        layers = []
        if expansion != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1,
                      groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # 最后不接激活函数：线性瓶颈保留低维信息。
        ])
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        return x + result if self.use_residual else result


class MobileNetV2(nn.Module):
    """默认七组 (扩张倍率, 通道, 重复数, 首块步长) 对应原论文。"""

    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0,
                 last_channel: int = 1280):
        super().__init__()
        if width_mult <= 0 or last_channel <= 0:
            raise ValueError("width_mult 与 last_channel 必须为正")
        stages = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2),
                  (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2),
                  (6, 320, 1, 1)]
        input_channels = _make_divisible(32 * width_mult)
        output_channels = _make_divisible(last_channel * max(1.0, width_mult))
        layers = [nn.Sequential(
            nn.Conv2d(3, input_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channels), nn.ReLU6(inplace=True),
        )]
        for expansion, channels, repeat, stride in stages:
            stage_channels = _make_divisible(channels * width_mult)
            for index in range(repeat):
                layers.append(InvertedResidual(
                    input_channels, stage_channels,
                    stride if index == 0 else 1, expansion,
                ))
                input_channels = stage_channels
        layers.append(nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels), nn.ReLU6(inplace=True),
        ))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)).flatten(1))


if __name__ == "__main__":
    model = MobileNetV2(num_classes=10, width_mult=0.25, last_channel=32)
    print(model(torch.randn(2, 3, 64, 64)).shape)
