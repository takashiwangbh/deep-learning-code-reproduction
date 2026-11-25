"""DDPM：逐步加高斯噪声、预测噪声、按后验分布反向采样。

Ho et al., Denoising Diffusion Probabilistic Models:
https://arxiv.org/abs/2006.11239

为了看清公式，本文件直接实现扩散过程。默认的小 U-Net 用于学习实验；论文级
图像质量还需要大模型、长时间训练和数据预处理。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """把整数时间步 t 编成连续向量；sin/cos 提供不同频率。"""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    phase = t.float()[:, None] * frequencies[None]
    embedding = torch.cat((phase.sin(), phase.cos()), dim=1)
    return F.pad(embedding, (0, dim - embedding.shape[1]))


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_projection(F.silu(time))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """两层降采样 + 跳跃连接；输入 H、W 必须是 4 的倍数。"""

    def __init__(self, image_channels: int = 3, base_channels: int = 32, time_dim: int = 128):
        super().__init__()
        c = base_channels
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.stem = nn.Conv2d(image_channels, c, 3, padding=1)
        self.down1 = ResBlock(c, c, time_dim)
        self.downsample1 = nn.Conv2d(c, 2 * c, 4, stride=2, padding=1)
        self.down2 = ResBlock(2 * c, 2 * c, time_dim)
        self.downsample2 = nn.Conv2d(2 * c, 4 * c, 4, stride=2, padding=1)
        self.middle = ResBlock(4 * c, 4 * c, time_dim)
        self.upsample2 = nn.ConvTranspose2d(4 * c, 2 * c, 4, stride=2, padding=1)
        self.up2 = ResBlock(4 * c, 2 * c, time_dim)
        self.upsample1 = nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1)
        self.up1 = ResBlock(2 * c, c, time_dim)
        self.out_norm = nn.GroupNorm(_group_count(c), c)
        self.out_conv = nn.Conv2d(c, image_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 4 or x.shape[-1] % 4:
            raise ValueError("image height and width must be divisible by 4")
        time = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        h1 = self.down1(self.stem(x), time)
        h2 = self.down2(self.downsample1(h1), time)
        middle = self.middle(self.downsample2(h2), time)
        up2 = self.up2(torch.cat((self.upsample2(middle), h2), dim=1), time)
        up1 = self.up1(torch.cat((self.upsample1(up2), h1), dim=1), time)
        return self.out_conv(F.silu(self.out_norm(up1)))


class DDPM(nn.Module):
    def __init__(
        self, denoiser: nn.Module, timesteps: int = 1000,
        beta_start: float = 1e-4, beta_end: float = 0.02,
        clip_denoised: bool = True,
    ):
        super().__init__()
        if timesteps < 2 or not (0 < beta_start < beta_end < 1):
            raise ValueError("require timesteps >= 2 and 0 < beta_start < beta_end < 1")
        self.denoiser = denoiser
        self.timesteps = timesteps
        # 像素图像可裁剪到 [-1,1]；潜空间一般无此界限，应设 False。
        self.clip_denoised = clip_denoised
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        # register_buffer 使调度表随 model.to(device) 移动，但不会被优化器更新。
        for name, tensor in (
            ("betas", betas), ("alphas", alphas), ("alpha_bars", alpha_bars),
            ("alpha_bars_prev", alpha_bars_prev),
        ):
            self.register_buffer(name, tensor)

    @staticmethod
    def _at(schedule: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return schedule[t].reshape(-1, 1, 1, 1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """q(x_t|x_0) = sqrt(alpha_bar_t)x_0 + sqrt(1-alpha_bar_t)epsilon。"""
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar = self._at(self.alpha_bars, t)
        return alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise

    def training_loss(self, x0: torch.Tensor, **model_kwargs: torch.Tensor) -> torch.Tensor:
        """随机选 t，再训练 epsilon_theta(x_t,t) 预测加入的 epsilon。"""
        t = torch.randint(self.timesteps, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        return F.mse_loss(self.denoiser(self.q_sample(x0, t, noise), t, **model_kwargs), noise)

    @torch.no_grad()
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, **model_kwargs: torch.Tensor) -> torch.Tensor:
        """从 p_theta(x_{t-1}|x_t) 采样；t=0 时返回预测的 x_0。"""
        alpha = self._at(self.alphas, t)
        beta = self._at(self.betas, t)
        alpha_bar = self._at(self.alpha_bars, t)
        alpha_bar_prev = self._at(self.alpha_bars_prev, t)
        predicted_noise = self.denoiser(xt, t, **model_kwargs)
        predicted_x0 = (xt - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
        if self.clip_denoised:
            predicted_x0 = predicted_x0.clamp(-1, 1)
        # q(x_{t-1}|x_t,x_0) 的均值与方差；替换其中的 x_0 为网络预测。
        mean = (
            alpha_bar_prev.sqrt() * beta / (1 - alpha_bar) * predicted_x0
            + alpha.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar) * xt
        )
        variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
        has_noise = (t > 0).reshape(-1, 1, 1, 1)
        return mean + has_noise * variance.clamp_min(0).sqrt() * torch.randn_like(xt)

    @torch.no_grad()
    def sample(
        self, shape: tuple[int, int, int, int], device: torch.device | str | None = None,
        **model_kwargs: torch.Tensor,
    ) -> torch.Tensor:
        if device is None:
            device = self.betas.device
        x = torch.randn(shape, device=device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x = self.p_sample(x, t, **model_kwargs)
        return x


if __name__ == "__main__":
    model = DDPM(TinyUNet(image_channels=1, base_channels=8), timesteps=8)
    images = torch.randn(2, 1, 16, 16).clamp(-1, 1)
    loss = model.training_loss(images)
    loss.backward()
    samples = model.sample((2, 1, 16, 16))
    print(f"loss={loss.item():.3f}, samples={tuple(samples.shape)}")
