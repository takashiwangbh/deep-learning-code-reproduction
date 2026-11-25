"""DDIM 跳步采样：eta=0 时给定初始噪声后完全确定。"""

import torch
from torch import nn
from torch.nn import functional as F


class TinyDenoiser(nn.Module):
    def __init__(self, channels=1, hidden=16, total_steps=20):
        super().__init__()
        self.total_steps = total_steps
        self.network = nn.Sequential(nn.Conv2d(channels + 1, hidden, 3, padding=1), nn.ReLU(), nn.Conv2d(hidden, channels, 3, padding=1))

    def forward(self, noisy, t):
        time_map = (t.float() / self.total_steps)[:, None, None, None].expand(-1, 1, *noisy.shape[-2:])
        return self.network(torch.cat((noisy, time_map), dim=1))


class DDIM(nn.Module):
    def __init__(self, total_steps=20):
        super().__init__()
        betas = torch.linspace(1e-4, 0.02, total_steps)
        self.register_buffer("alpha_bars", (1 - betas).cumprod(0))

    def q_sample(self, clean, t, noise):
        alpha_bar = self.alpha_bars[t].reshape(-1, *([1] * (clean.ndim - 1)))
        return alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise

    def training_loss(self, denoiser, clean):
        batch = clean.shape[0]
        t = torch.randint(len(self.alpha_bars), (batch,), device=clean.device)
        noise = torch.randn_like(clean)
        return F.mse_loss(denoiser(self.q_sample(clean, t, noise), t), noise)

    def step(self, noisy, predicted_noise, t, previous_t, eta=0.0):
        if not 0 <= eta <= 1:
            raise ValueError("eta must be between 0 and 1")
        a_t = self.alpha_bars[t]
        a_prev = self.alpha_bars[previous_t] if previous_t >= 0 else a_t.new_ones(())
        estimated_clean = (noisy - (1 - a_t).sqrt() * predicted_noise) / a_t.sqrt()
        sigma = eta * (((1 - a_prev) / (1 - a_t)) * (1 - a_t / a_prev)).clamp_min(0).sqrt()
        direction = (1 - a_prev - sigma.square()).clamp_min(0).sqrt() * predicted_noise
        previous = a_prev.sqrt() * estimated_clean + direction
        if eta > 0 and previous_t >= 0:
            previous = previous + sigma * torch.randn_like(noisy)
        return previous

    @torch.no_grad()
    def sample(self, denoiser, initial_noise, inference_steps=5, eta=0.0):
        if not 1 <= inference_steps <= len(self.alpha_bars):
            raise ValueError("inference_steps must be between 1 and total_steps")
        times = torch.linspace(len(self.alpha_bars) - 1, 0, inference_steps).round().long().tolist()
        current = initial_noise.clone()
        for index, t in enumerate(times):
            previous_t = times[index + 1] if index + 1 < len(times) else -1
            time_batch = torch.full((current.shape[0],), t, device=current.device, dtype=torch.long)
            current = self.step(current, denoiser(current, time_batch), t, previous_t, eta)
        return current


if __name__ == "__main__":
    torch.manual_seed(0)
    schedule, denoiser = DDIM(), TinyDenoiser()
    clean = torch.randn(2, 1, 8, 8)
    loss = schedule.training_loss(denoiser, clean)
    loss.backward()
    noise = torch.randn_like(clean)
    assert torch.equal(schedule.sample(denoiser, noise), schedule.sample(denoiser, noise))
    print("sample", tuple(schedule.sample(denoiser, noise).shape), "training loss", round(loss.item(), 3))
