"""FPN: lateral 1x1 projections and top-down multi-scale semantic fusion."""

import torch
from torch import nn
from torch.nn import functional as F


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, input_channels=(32, 64, 128, 256), width: int = 64):
        super().__init__()
        if len(input_channels) != 4:
            raise ValueError("Expected C2, C3, C4, C5")
        self.laterals = nn.ModuleList(nn.Conv2d(ch, width, 1) for ch in input_channels)
        self.smoothing = nn.ModuleList(nn.Conv2d(width, width, 3, padding=1) for _ in range(4))

    def forward(self, features: list[torch.Tensor]):
        """C2..C5 -> P2..P6; P2 has finest spatial resolution."""
        if len(features) != 4:
            raise ValueError("Expected four backbone feature maps")
        lateral = [conv(x) for conv, x in zip(self.laterals, features)]
        merged = [None] * 4
        merged[-1] = lateral[-1]
        for level in (2, 1, 0):
            merged[level] = lateral[level] + F.interpolate(
                merged[level + 1], size=lateral[level].shape[-2:], mode="nearest")
        outputs = [smooth(x) for smooth, x in zip(self.smoothing, merged)]
        return {**{f"P{level}": feature for level, feature in enumerate(outputs, start=2)},
                "P6": F.max_pool2d(outputs[-1], 1, stride=2)}


class TinyBackbone(nn.Module):
    """Small local source of C2..C5 for the runnable example."""

    def __init__(self, channels=(32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, channels[0], 3, stride=2, padding=1), nn.ReLU())
        stages, current = [], channels[0]
        for width in channels:
            stages.append(nn.Sequential(nn.Conv2d(current, width, 3, stride=2, padding=1), nn.ReLU(),
                                         nn.Conv2d(width, width, 3, padding=1), nn.ReLU()))
            current = width
        self.stages = nn.ModuleList(stages)

    def forward(self, image):
        image = self.stem(image)
        outputs = []
        for stage in self.stages:
            image = stage(image)
            outputs.append(image)
        return outputs


if __name__ == "__main__":
    torch.manual_seed(0)
    backbone = TinyBackbone(channels=(8, 16, 24, 32))
    pyramid = FeaturePyramidNetwork((8, 16, 24, 32), width=12)
    maps = pyramid(backbone(torch.randn(2, 3, 64, 64)))
    sum(x.mean() for x in maps.values()).backward()
    print("FPN feature maps:", {name: tuple(tensor.shape) for name, tensor in maps.items()})
