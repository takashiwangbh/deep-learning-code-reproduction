"""LeNet-5：卷积、池化和全连接分类器的入门实现。"""

import torch
from torch import nn


class LeNet5(nn.Module):
    """输入 [B, 1, 32, 32]，输出 [B, num_classes] 的 logits。

    原论文的 C3 使用部分通道连接，S 层有可学习系数。这里使用完整连接的
    Conv2d 和固定平均池化，保留 LeNet-5 的尺寸与学习路径，更便于初学者阅读。
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),  # 32 -> 28: C1
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),  # 28 -> 14: S2
            nn.Conv2d(6, 16, kernel_size=5),  # 14 -> 10: C3
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),  # 10 -> 5: S4
            nn.Conv2d(16, 120, kernel_size=5),  # 5 -> 1: C5
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),  # F6
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (1, 32, 32):
            raise ValueError("LeNet5 需要 [B, 1, 32, 32] 输入")
        return self.classifier(self.features(x))


if __name__ == "__main__":
    model = LeNet5()
    print(model(torch.randn(2, 1, 32, 32)).shape)
