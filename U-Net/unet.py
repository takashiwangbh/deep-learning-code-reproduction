"""U-Net：编码器到解码器的同分辨率跳接，逐像素预测。"""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    """连续两个 3×3 卷积；padding=1 让输出尺寸与输入一致。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(skip_channels + out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 输入高宽不是 16 的倍数时补齐边界，仍可拼接编码器特征。
        diff_h, diff_w = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                      diff_h // 2, diff_h - diff_h // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """输出 [B, num_classes, H, W] 的像素 logits，不先做 softmax。

    原论文用无填充的 valid 卷积并裁剪 skip；这里采用常见的 same-size
    padding 变体，便于初学者对齐像素标签，主干拓扑保持一致。
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 base_channels: int = 64):
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels 必须为正")
        c = base_channels
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.bottom = DoubleConv(c * 8, c * 16)
        self.down = nn.MaxPool2d(2)
        self.up4 = Up(c * 16, c * 8, c * 8)
        self.up3 = Up(c * 8, c * 4, c * 4)
        self.up2 = Up(c * 4, c * 2, c * 2)
        self.up1 = Up(c * 2, c, c)
        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        e4 = self.enc4(self.down(e3))
        x = self.bottom(self.down(e4))
        x = self.up4(x, e4)
        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)
        return self.head(x)


if __name__ == "__main__":
    model = UNet(num_classes=2, base_channels=8)
    print(model(torch.randn(2, 3, 65, 67)).shape)
