"""FCN-8s 的多级 skip 融合：pool5 -> pool4 -> pool3 -> 原图。"""
import torch
from torch import nn
from torch.nn import functional as F


def conv_stage(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
    )


class FCN8s(nn.Module):
    def __init__(self, classes=3, width=16):
        super().__init__()
        widths = [width, width * 2, width * 4, width * 4, width * 4]
        previous = 3
        self.stages = nn.ModuleList()
        for channels in widths:
            self.stages.append(conv_stage(previous, channels))
            previous = channels
        self.score3 = nn.Conv2d(widths[2], classes, 1)
        self.score4 = nn.Conv2d(widths[3], classes, 1)
        self.score5 = nn.Conv2d(widths[4], classes, 1)

    def forward(self, image):
        h, w = image.shape[-2:]
        x = image
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        pool3, pool4, pool5 = features[2:]
        score = F.interpolate(self.score5(pool5), size=pool4.shape[-2:], mode="bilinear", align_corners=False)
        score = score + self.score4(pool4)
        score = F.interpolate(score, size=pool3.shape[-2:], mode="bilinear", align_corners=False)
        score = score + self.score3(pool3)
        return F.interpolate(score, size=(h, w), mode="bilinear", align_corners=False)


if __name__ == "__main__":
    model = FCN8s()
    logits = model(torch.randn(2, 3, 64, 64))
    logits.sum().backward()
    print("FCN", tuple(logits.shape))
