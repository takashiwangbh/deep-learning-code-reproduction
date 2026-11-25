"""SegNet：编码器保存最大池化位置，解码器按原位置反池化。"""
import torch
from torch import nn
from torch.nn import functional as F


class SegNet(nn.Module):
    def __init__(self, classes=3, width=16):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(width, 2 * width, 3, padding=1), nn.BatchNorm2d(2 * width), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(2 * width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU())
        self.classifier = nn.Conv2d(width, classes, 1)

    def forward(self, image):
        e1 = self.enc1(image)
        p1, index1 = F.max_pool2d(e1, 2, return_indices=True)
        e2 = self.enc2(p1)
        p2, index2 = F.max_pool2d(e2, 2, return_indices=True)
        x = F.max_unpool2d(p2, index2, 2, output_size=e2.shape)
        x = self.dec2(x)
        x = F.max_unpool2d(x, index1, 2, output_size=e1.shape)
        return self.classifier(self.dec1(x))


if __name__ == "__main__":
    model = SegNet()
    logits = model(torch.randn(2, 3, 32, 32))
    logits.sum().backward()
    print("SegNet", tuple(logits.shape))
