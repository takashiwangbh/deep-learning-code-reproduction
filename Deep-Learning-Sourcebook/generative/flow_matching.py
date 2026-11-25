"""直线路径的条件 Flow Matching：学习速度场并积分生成样本。

Lipman et al., Flow Matching for Generative Modeling:
https://arxiv.org/abs/2210.02747

此处采用最容易理解的独立噪声/数据直线配对；不包含论文的所有概率路径。
"""

import torch
from torch import nn
from torch.nn import functional as F

from ddpm import TinyUNet  # 复用本目录内可读的时间条件网络源码


class FlowMatching(nn.Module):
    def __init__(self, velocity_model: nn.Module):
        super().__init__()
        self.velocity_model = velocity_model

    @staticmethod
    def interpolation(noise: torch.Tensor, data: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x_t=(1-t)x_0+t x_1；这条路径的目标速度是 x_1-x_0。"""
        if noise.shape != data.shape:
            raise ValueError("noise and data must have the same shape")
        t = t.reshape((-1,) + (1,) * (data.ndim - 1))
        return (1 - t) * noise + t * data, data - noise

    def training_loss(self, data: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(data)
        t = torch.rand(data.size(0), device=data.device)
        xt, target_velocity = self.interpolation(noise, data, t)
        return F.mse_loss(self.velocity_model(xt, t), target_velocity)

    @torch.no_grad()
    def sample(
        self, shape: tuple[int, ...], steps: int = 32,
        device: torch.device | str | None = None, initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """显式 Euler：x_{t+Δt}=x_t+Δt v_theta(x_t,t)。"""
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if device is None:
            device = next(self.velocity_model.parameters(), torch.empty(0)).device
        x = torch.randn(shape, device=device) if initial_noise is None else initial_noise.clone()
        if tuple(x.shape) != shape:
            raise ValueError("initial_noise shape must equal shape")
        for index in range(steps):
            t = torch.full((shape[0],), index / steps, device=x.device)
            x = x + self.velocity_model(x, t) / steps
        return x


if __name__ == "__main__":
    model = FlowMatching(TinyUNet(image_channels=1, base_channels=8))
    images = torch.randn(2, 1, 8, 8).clamp(-1, 1)
    loss = model.training_loss(images)
    loss.backward()
    samples = model.sample((2, 1, 8, 8), steps=4)
    print(f"loss={loss.item():.3f}, samples={tuple(samples.shape)}")
