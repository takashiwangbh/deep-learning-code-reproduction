"""AlexNet：较大卷积核、ReLU、LRN 和 dropout 的经典图像分类网络。"""

import torch
from torch import nn


def _even_channels(original: int, width_mult: float) -> int:
    # 分组卷积需要输入/输出通道数可被 2 整除。
    return max(2, 2 * round(original * width_mult / 2))


class AlexNet(nn.Module):
    """单机 PyTorch 复现；groups=2 表达原论文跨两块 GPU 的分组连接。

    `width_mult=1, hidden=4096, pool_size=6` 接近经典 2012 配置。
    缩小这些参数可以在 CPU 上快速学习与测试；网络拓扑不变。
    """

    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        hidden: int = 4096,
        pool_size: int = 6,
    ):
        super().__init__()
        if width_mult <= 0 or hidden <= 0 or pool_size <= 0:
            raise ValueError("width_mult、hidden 和 pool_size 必须为正")
        c1, c2, c3, c4, c5 = (
            _even_channels(c, width_mult) for c in (96, 256, 384, 384, 256)
        )
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(c1, c2, 5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, 3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c5, 3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(c5 * pool_size * pool_size, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.avgpool(self.features(x)))


if __name__ == "__main__":
    model = AlexNet(width_mult=0.25, hidden=64, pool_size=1, num_classes=10)
    print(model(torch.randn(2, 3, 96, 96)).shape)
