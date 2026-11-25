"""SimCLR：同图双随机视图、共享编码器、非线性投影头与 NT-Xent。

论文：https://arxiv.org/abs/2002.05709
本文件只依赖 PyTorch；使用小型 ResNet-18 风格编码器便于 CPU 学习。
论文主要实验采用 ResNet-50、大批量和更完整的数据增强，不能由此示例复现指标。
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def augment_view(images: torch.Tensor) -> torch.Tensor:
    """对 [N,3,H,W] 的 [0,1] 图像做随机裁剪、翻转、颜色扰动和模糊。

    每次调用独立采样；`make_two_views` 连续调用两次，得到正样本对。
    这里是可读的单机增强子集，省略论文的全部概率与参数配方。
    """
    if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
        raise ValueError("augment_view 需要浮点 RGB 图像 [N,3,H,W]")
    _, _, height, width = images.shape
    results = []
    for image in images:
        scale = 0.5 + 0.5 * torch.rand(()).item()
        crop_h = max(1, round(height * math.sqrt(scale)))
        crop_w = max(1, round(width * math.sqrt(scale)))
        top = torch.randint(height - crop_h + 1, ()).item()
        left = torch.randint(width - crop_w + 1, ()).item()
        crop = image[:, top:top + crop_h, left:left + crop_w][None]
        view = F.interpolate(crop, size=(height, width), mode="bilinear",
                             align_corners=False)
        if torch.rand(()).item() < 0.5:
            view = view.flip(-1)

        # 亮度、对比度、饱和度的轻量实现；保持输入尺寸和 RGB 三通道。
        brightness = 0.6 + 0.8 * torch.rand(()).item()
        contrast = 0.6 + 0.8 * torch.rand(()).item()
        saturation = 0.6 + 0.8 * torch.rand(()).item()
        view = view * brightness
        view = (view - view.mean(dim=(-2, -1), keepdim=True)) * contrast + view.mean(
            dim=(-2, -1), keepdim=True
        )
        gray = (view[:, 0:1] * 0.299 + view[:, 1:2] * 0.587
                + view[:, 2:3] * 0.114)
        view = gray + saturation * (view - gray)

        if torch.rand(()).item() < 0.5:
            sigma = 0.3 + 1.2 * torch.rand(()).item()
            axis = torch.arange(-2, 3, device=view.device, dtype=view.dtype)
            kernel_1d = torch.exp(-axis.square() / (2 * sigma * sigma))
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
            weight = kernel_2d[None, None].expand(3, 1, 5, 5)
            view = F.conv2d(view, weight, padding=2, groups=3)
        results.append(view[0].clamp(0, 1))
    return torch.stack(results, dim=0)


def make_two_views(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """同一批无标签图像经过两次独立增强；索引 i 对应同一原图。"""
    return augment_view(images), augment_view(images)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (nn.Identity() if stride == 1 and in_channels == out_channels
                     else nn.Sequential(
                         nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                         nn.BatchNorm2d(out_channels)))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.main(x) + self.skip(x))


class SmallResNetEncoder(nn.Module):
    """四组各两块的 ResNet-18 风格特征提取器，输出平均池化后的 h。"""

    def __init__(self, width: int = 64):
        super().__init__()
        if width <= 0:
            raise ValueError("width 必须为正")
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        )
        blocks = []
        in_channels = width
        for stage, out_channels in enumerate((width, width * 2, width * 4, width * 8)):
            blocks.append(ResidualBlock(in_channels, out_channels,
                                        stride=1 if stage == 0 else 2))
            blocks.append(ResidualBlock(out_channels, out_channels))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = in_channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.pool(self.blocks(self.stem(images))).flatten(1)


class SimCLR(nn.Module):
    """共享编码器 f 和两层 MLP g；训练用 z，迁移学习用 h=f(image)。"""

    def __init__(self, width: int = 64, projection_hidden: int = 2048,
                 projection_dim: int = 128):
        super().__init__()
        if projection_hidden <= 0 or projection_dim <= 0:
            raise ValueError("投影层尺寸必须为正")
        self.encoder = SmallResNetEncoder(width)
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.output_dim, projection_hidden),
            nn.ReLU(inplace=True), nn.Linear(projection_hidden, projection_dim),
        )

    def forward(self, view1: torch.Tensor, view2: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if view1.shape != view2.shape or view1.ndim != 4:
            raise ValueError("两组视图必须同形且为 [N,3,H,W]")
        # 一起编码，两视图共享网络权重，BatchNorm 也看到同一 2N 批次。
        h = self.encoder(torch.cat((view1, view2), dim=0))
        z = self.projector(h)
        return z.chunk(2, dim=0)

    def representation(self, images: torch.Tensor) -> torch.Tensor:
        """预训练后常保留 h，而丢弃只服务于对比损失的投影头 g。"""
        return self.encoder(images)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.5) -> torch.Tensor:
    """2N 个样本的对称 NT-Xent；第 i 个 view1 与第 i 个 view2 为正对。

    对每个锚点，分母包含它之外的 2N-1 个向量（包括正例），排除自身；
    其余 2N-2 个是批内负例。余弦相似度用 L2 归一化和矩阵乘法实现。
    """
    if z1.ndim != 2 or z1.shape != z2.shape or z1.shape[0] < 2:
        raise ValueError("z1/z2 需要同形 [N,D] 且 N>=2")
    if temperature <= 0:
        raise ValueError("temperature 必须为正")
    n = z1.shape[0]
    z = F.normalize(torch.cat((z1, z2), dim=0), dim=-1)
    similarity = z @ z.T / temperature
    indices = torch.arange(2 * n, device=z.device)
    positive_index = (indices + n) % (2 * n)
    # -inf 从 logsumexp 分母中排除自身；仍保留正例。
    candidates = similarity.masked_fill(torch.eye(2 * n, device=z.device,
                                                   dtype=torch.bool), -torch.inf)
    return (torch.logsumexp(candidates, dim=1)
            - similarity[indices, positive_index]).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    images = torch.rand(2, 3, 32, 32)
    view1, view2 = make_two_views(images)
    model = SimCLR(width=8, projection_hidden=32, projection_dim=16)
    z1, z2 = model(view1, view2)
    loss = nt_xent_loss(z1, z2)
    loss.backward()
    print("projections:", tuple(z1.shape), tuple(z2.shape),
          "NT-Xent:", round(loss.item(), 4))
