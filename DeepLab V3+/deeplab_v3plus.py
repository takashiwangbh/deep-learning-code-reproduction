"""DeepLabv3+：空洞空间金字塔池化 ASPP + 浅层特征解码器。"""
import torch
from torch import nn
from torch.nn import functional as F


class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=dilation,
                                   dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.norm(self.pointwise(self.depthwise(x))))


class ASPP(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.direct = nn.Sequential(nn.Conv2d(channels, width, 1), nn.ReLU())
        self.rates = nn.ModuleList(SeparableConv(channels, width, r) for r in (1, 2, 3))
        self.global_project = nn.Conv2d(channels, width, 1)
        self.project = nn.Sequential(nn.Conv2d(5 * width, width, 1), nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        global_context = F.interpolate(
            self.global_project(F.adaptive_avg_pool2d(x, 1)), size=size, mode="bilinear", align_corners=False
        )
        return self.project(torch.cat([self.direct(x), *(branch(x) for branch in self.rates), global_context], 1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, classes=3, width=16):
        super().__init__()
        self.low = nn.Sequential(SeparableConv(3, width), nn.MaxPool2d(2),
                                 SeparableConv(width, width), nn.MaxPool2d(2))
        self.high = nn.Sequential(SeparableConv(width, 2 * width), nn.MaxPool2d(2),
                                  SeparableConv(2 * width, 4 * width), nn.MaxPool2d(2))
        self.aspp = ASPP(4 * width, width)
        self.low_project = nn.Conv2d(width, width // 2, 1)
        self.decoder = nn.Sequential(SeparableConv(width + width // 2, width),
                                     SeparableConv(width, width), nn.Conv2d(width, classes, 1))

    def forward(self, image):
        size = image.shape[-2:]
        low = self.low(image)
        high = self.aspp(self.high(low))
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.decoder(torch.cat([high, self.low_project(low)], 1))
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


if __name__ == "__main__":
    model = DeepLabV3Plus()
    logits = model(torch.randn(2, 3, 64, 64))
    logits.sum().backward()
    print("DeepLabv3+", tuple(logits.shape))
