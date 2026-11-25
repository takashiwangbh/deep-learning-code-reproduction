"""LoRA：冻结预训练线性层，只训练低秩矩阵 A、B。

W' = W + (alpha / rank) * B @ A；这里把乘法拆成两次 F.linear。
仓库根目录运行：python LoRA/lora.py
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """可替换普通 nn.Linear 的适配器，输入最后一维为 in_features。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 8.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0 or rank > min(in_features, out_features):
            raise ValueError("rank 应在 1 与 min(in_features, out_features) 之间")
        self.rank = rank
        self.scale = alpha / rank
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B=0 使适配器初始时与原线性层完全相同。
        # 作为持久 buffer 保存，避免 merge 后存档/加载时重复叠加 BA。
        self.register_buffer("_merged", torch.tensor(False))

    @property
    def merged(self) -> bool:
        return bool(self._merged.item())

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 4, alpha: float = 8.0) -> "LoRALinear":
        """用已训练的线性层初始化；不会修改传入的 linear。"""
        module = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            bias=linear.bias is not None,
        )
        with torch.no_grad():
            module.base.weight.copy_(linear.weight)
            if linear.bias is not None:
                module.base.bias.copy_(linear.bias)
        return module.to(device=linear.weight.device, dtype=linear.weight.dtype)

    def forward(self, x: Tensor) -> Tensor:
        output = self.base(x)
        if self.merged:
            return output
        return output + self.scale * F.linear(F.linear(x, self.A), self.B)

    @torch.no_grad()
    def merge(self) -> None:
        """推理时把 BA 合到 W，消除额外两次线性变换。"""
        if not self.merged:
            self.base.weight.add_(self.scale * (self.B @ self.A))
            self._merged.fill_(True)

    @torch.no_grad()
    def unmerge(self) -> None:
        """恢复原始 W，继续训练适配器。"""
        if self.merged:
            self.base.weight.sub_(self.scale * (self.B @ self.A))
            self._merged.fill_(False)


if __name__ == "__main__":
    original = nn.Linear(8, 6)
    adapter = LoRALinear.from_linear(original, rank=2)
    x = torch.randn(3, 8)
    print("初始最大差值:", (original(x) - adapter(x)).abs().max().item())
    print("可训练参数:", [name for name, p in adapter.named_parameters() if p.requires_grad])
