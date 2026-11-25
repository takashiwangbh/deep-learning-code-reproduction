"""批归一化和层归一化的教学实现；仅用 torch 张量运算。"""

import torch
from torch import nn


class BatchNorm1dFromScratch(nn.Module):
    """输入 [B, D]。训练时按 batch 统计，推理时使用移动平均。"""

    def __init__(self, features: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.register_buffer("running_mean", torch.zeros(features))
        self.register_buffer("running_var", torch.ones(features))
        self.momentum, self.eps = momentum, eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("BatchNorm1dFromScratch expects [batch, features]")
        if self.training:
            mean = x.mean(dim=0)
            var = ((x - mean) ** 2).mean(dim=0)
            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var, self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class LayerNormFromScratch(nn.Module):
    """输入 [..., D]，每个样本单独统计最后一维，训练/推理公式相同。"""

    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


if __name__ == "__main__":
    x = torch.randn(4, 8, requires_grad=True)
    batch_norm = BatchNorm1dFromScratch(8)
    layer_norm = LayerNormFromScratch(8)
    output = batch_norm(x) + layer_norm(x)
    output.square().mean().backward()
    print("normalized:", tuple(output.shape), "gradient:", x.grad is not None)
