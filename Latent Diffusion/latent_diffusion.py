"""潜空间扩散（LDM）的两个阶段与显式文本交叉注意力。

Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models:
https://arxiv.org/abs/2112.10752

1. 先训练卷积 VAE，把图像压缩成空间潜变量 z。
2. 冻结 VAE，在 z 上运行 DDPM；潜变量 U-Net 用 cross-attention 读取文本 token。

教学简化：文本编码器是从零训练的 token/位置嵌入，而不是预训练 CLIP；VAE
只用像素 MSE + KL，没有感知损失/对抗损失；U-Net 只有一个交叉注意力层；
没有 classifier-free guidance、复杂噪声调度或论文规模的训练数据与权重。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ddpm import DDPM, ResBlock, sinusoidal_embedding


def _groups(channels: int) -> int:
    for count in (8, 4, 2):
        if channels % count == 0:
            return count
    return 1


class SpatialVAE(nn.Module):
    """图像 [B,C,H,W] ↔ 空间潜变量 [B,Z,H/4,W/4]。"""

    def __init__(self, image_channels: int = 3, latent_channels: int = 4, base_channels: int = 32):
        super().__init__()
        if min(image_channels, latent_channels, base_channels) <= 0:
            raise ValueError("通道数必须为正")
        c = base_channels
        self.image_channels = image_channels
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, c, 3, padding=1), nn.SiLU(),
            nn.Conv2d(c, 2 * c, 4, stride=2, padding=1),
            nn.GroupNorm(_groups(2 * c), 2 * c), nn.SiLU(),
            nn.Conv2d(2 * c, 4 * c, 4, stride=2, padding=1),
            nn.GroupNorm(_groups(4 * c), 4 * c), nn.SiLU(),
            nn.Conv2d(4 * c, 2 * latent_channels, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 4 * c, 3, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(4 * c, 2 * c, 4, stride=2, padding=1),
            nn.GroupNorm(_groups(2 * c), 2 * c), nn.SiLU(),
            nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1),
            nn.GroupNorm(_groups(c), c), nn.SiLU(),
            nn.Conv2d(c, image_channels, 3, padding=1), nn.Tanh(),
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != self.image_channels:
            raise ValueError("images 需要 [B,image_channels,H,W]")
        if images.shape[-2] % 4 or images.shape[-1] % 4:
            raise ValueError("H/W 必须能被 4 整除")
        return self.encoder(images).chunk(2, dim=1)  # μ 与 log σ² 都是空间特征图

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # 与 vae.py 的向量 VAE 相同，只是 μ/log σ² 现在有空间维度。
        return mu + (0.5 * logvar).exp() * torch.randn_like(mu)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(images)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    def loss(
        self, images: torch.Tensor, kl_weight: float = 1e-4
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """教学用第一阶段目标：像素 MSE + 小权重 KL。图像应在 [-1,1]。"""
        reconstruction, mu, logvar = self(images)
        pixel_loss = F.mse_loss(reconstruction, images)
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum() / images.size(0)
        return pixel_loss + kl_weight * kl, pixel_loss, kl


class TokenConditioner(nn.Module):
    """把整数 token IDs 映射为一组条件向量；0 默认为 PAD。"""

    def __init__(self, vocab_size: int, context_dim: int, max_length: int = 32, pad_id: int = 0):
        super().__init__()
        if vocab_size <= 1 or context_dim <= 0 or max_length <= 0:
            raise ValueError("词表、条件维度和最大长度必须为正")
        self.max_length = max_length
        self.pad_id = pad_id
        self.token = nn.Embedding(vocab_size, context_dim, padding_idx=pad_id)
        self.position = nn.Embedding(max_length, context_dim)
        self.norm = nn.LayerNorm(context_dim)

    def forward(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 2 or not 0 < token_ids.size(1) <= self.max_length:
            raise ValueError("token_ids 需要非空 [B,T]，且 T 不超过 max_length")
        if attention_mask is None:
            attention_mask = token_ids.ne(self.pad_id)
        if attention_mask.shape != token_ids.shape:
            raise ValueError("attention_mask 形状必须与 token_ids 相同")
        attention_mask = attention_mask.bool()
        if not bool(attention_mask.any(dim=1).all()):
            raise ValueError("每个文本至少需要一个有效 token")
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        context = self.norm(self.token(token_ids) + self.position(positions))
        return context, attention_mask


class CrossAttention2d(nn.Module):
    """潜图每个空间位置作 Q；文本 token 作 K/V。"""

    def __init__(self, channels: int, context_dim: int, heads: int = 4):
        super().__init__()
        if channels % heads:
            raise ValueError("channels 必须能被 heads 整除")
        self.heads = heads
        self.head_dim = channels // heads
        self.norm = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(context_dim, channels)
        self.to_v = nn.Linear(context_dim, channels)
        self.out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if context.ndim != 3 or context.size(0) != batch or mask.shape != context.shape[:2]:
            raise ValueError("context 需要 [B,T,D]，mask 需要 [B,T]")
        image_tokens = x.flatten(start_dim=2).transpose(1, 2)  # [B,H*W,C]
        query = self.to_q(self.norm(image_tokens))
        key, value = self.to_k(context), self.to_v(context)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, tensor.size(1), self.heads, self.head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (query, key, value))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attended = scores.softmax(dim=-1) @ v
        attended = attended.transpose(1, 2).reshape(batch, height * width, channels)
        return x + self.out(attended).transpose(1, 2).reshape(batch, channels, height, width)


class ConditionalLatentUNet(nn.Module):
    """带时间条件与文本交叉注意力的两级潜空间 U-Net。"""

    def __init__(
        self, latent_channels: int = 4, base_channels: int = 32,
        context_dim: int = 64, heads: int = 4, time_dim: int = 128,
    ):
        super().__init__()
        c = base_channels
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.stem = nn.Conv2d(latent_channels, c, 3, padding=1)
        self.down_block = ResBlock(c, c, time_dim)
        self.downsample = nn.Conv2d(c, 2 * c, 4, stride=2, padding=1)
        self.middle = ResBlock(2 * c, 2 * c, time_dim)
        self.cross_attention = CrossAttention2d(2 * c, context_dim, heads)
        self.upsample = nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1)
        self.up_block = ResBlock(2 * c, c, time_dim)
        self.out_norm = nn.GroupNorm(_groups(c), c)
        self.out = nn.Conv2d(c, latent_channels, 3, padding=1)

    def forward(
        self, latent: torch.Tensor, t: torch.Tensor,
        context: torch.Tensor, context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if latent.shape[-2] % 2 or latent.shape[-1] % 2:
            raise ValueError("潜变量 H/W 必须能被 2 整除")
        time = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        skip = self.down_block(self.stem(latent), time)
        middle = self.middle(self.downsample(skip), time)
        middle = self.cross_attention(middle, context, context_mask)
        up = self.up_block(torch.cat((self.upsample(middle), skip), dim=1), time)
        return self.out(F.silu(self.out_norm(up)))


class LatentDiffusion(nn.Module):
    """完整教学路径：图像 ⇄ 空间潜变量，潜变量中的条件扩散。"""

    def __init__(
        self, image_size: int = 32, image_channels: int = 3, latent_channels: int = 4,
        base_channels: int = 32, vocab_size: int = 1000, context_dim: int = 64,
        max_text_length: int = 32, heads: int = 4, timesteps: int = 1000,
        latent_scale: float = 1.0,
    ):
        super().__init__()
        if image_size <= 0 or image_size % 8:
            raise ValueError("image_size 必须为 8 的正整数倍")
        if latent_scale <= 0:
            raise ValueError("latent_scale 必须为正")
        self.image_size = image_size
        self.image_channels = image_channels
        self.latent_channels = latent_channels
        self.register_buffer("latent_scale", torch.tensor(float(latent_scale)))
        self.autoencoder = SpatialVAE(image_channels, latent_channels, base_channels)
        self.conditioner = TokenConditioner(vocab_size, context_dim, max_text_length)
        denoiser = ConditionalLatentUNet(latent_channels, base_channels, context_dim, heads)
        # z 的值域不受 [-1,1] 约束，因此不能套用像素扩散时的 x0 裁剪。
        self.diffusion = DDPM(denoiser, timesteps=timesteps, clip_denoised=False)

    def autoencoder_loss(self, images: torch.Tensor, kl_weight: float = 1e-4):
        """第一阶段：仅优化 self.autoencoder.parameters()。"""
        self._check_images(images)
        return self.autoencoder.loss(images, kl_weight)

    def freeze_autoencoder(self) -> None:
        """第一阶段训练完成后调用，再优化 conditioner + diffusion。"""
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()

    def _check_images(self, images: torch.Tensor) -> None:
        if images.ndim != 4 or images.shape[1:] != (
            self.image_channels, self.image_size, self.image_size
        ):
            raise ValueError("images 形状必须匹配构造时设置的通道数和边长")

    def training_loss(
        self, images: torch.Tensor, token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """第二阶段：编码图像后在潜变量上预测扩散噪声。"""
        self._check_images(images)
        with torch.no_grad():
            mu, logvar = self.autoencoder.encode(images)
            latent = self.autoencoder.reparameterize(mu, logvar) * self.latent_scale
        context, mask = self.conditioner(token_ids, attention_mask)
        if context.size(0) != images.size(0):
            raise ValueError("图像与文本 batch 大小必须一致")
        return self.diffusion.training_loss(latent, context=context, context_mask=mask)

    @torch.no_grad()
    def sample(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """从随机潜变量采样，最后由冻结的 VAE 解码回图像。"""
        context, mask = self.conditioner(token_ids, attention_mask)
        batch = token_ids.size(0)
        side = self.image_size // 4
        latent = self.diffusion.sample(
            (batch, self.latent_channels, side, side), device=token_ids.device,
            context=context, context_mask=mask,
        )
        return self.autoencoder.decode(latent / self.latent_scale)


if __name__ == "__main__":
    torch.set_num_threads(1)
    model = LatentDiffusion(
        image_size=16, image_channels=1, latent_channels=2,
        base_channels=8, vocab_size=16, context_dim=16,
        max_text_length=4, heads=2, timesteps=4,
    )
    images = torch.randn(2, 1, 16, 16).clamp(-1, 1)
    tokens = torch.tensor([[1, 2, 0], [3, 4, 5]])
    ae_loss, _, _ = model.autoencoder_loss(images)
    ae_loss.backward()  # 演示第一阶段的可微损失；真实任务需先充分训练 VAE
    model.zero_grad(set_to_none=True)
    model.freeze_autoencoder()
    diffusion_loss = model.training_loss(images, tokens)
    diffusion_loss.backward()
    generated = model.sample(tokens)
    print(f"VAE={ae_loss.item():.3f}, DDPM={diffusion_loss.item():.3f}, image={tuple(generated.shape)}")
