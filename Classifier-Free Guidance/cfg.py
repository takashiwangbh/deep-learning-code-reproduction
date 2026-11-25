"""Classifier-free guidance: conditional dropout in training, dual prediction in sampling."""

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalNoisePredictor(nn.Module):
    def __init__(self, dimensions: int = 4, classes: int = 3, hidden: int = 64):
        super().__init__()
        self.null_condition = classes
        self.condition = nn.Embedding(classes + 1, hidden)
        self.input = nn.Linear(dimensions, hidden)
        self.time = nn.Linear(1, hidden)
        self.output = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                    nn.Linear(hidden, dimensions))

    def forward(self, noisy: torch.Tensor, time: torch.Tensor, labels: torch.Tensor | None):
        if labels is None:
            labels = torch.full((noisy.shape[0],), self.null_condition,
                                dtype=torch.long, device=noisy.device)
        hidden = self.input(noisy) + self.time(time.float()[:, None]) + self.condition(labels)
        return self.output(hidden)


def drop_conditions(labels: torch.Tensor, null_condition: int, probability: float):
    if not 0 <= probability <= 1:
        raise ValueError("Drop probability must be in [0,1]")
    mask = torch.rand(labels.shape, device=labels.device) < probability
    return torch.where(mask, null_condition, labels), mask


def guided_prediction(model: ConditionalNoisePredictor, noisy: torch.Tensor,
                      time: torch.Tensor, labels: torch.Tensor, scale: float = 3.0):
    """scale=0 is unconditional; scale=1 is conditional prediction."""
    unconditional = model(noisy, time, None)
    conditional = model(noisy, time, labels)
    return unconditional + scale * (conditional - unconditional)


class GuidedDiffusion(nn.Module):
    def __init__(self, dimensions: int = 4, classes: int = 3, steps: int = 20):
        super().__init__()
        self.predictor = ConditionalNoisePredictor(dimensions, classes)
        betas = torch.linspace(1e-4, 0.02, steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("alpha_bars", (1 - betas).cumprod(dim=0))

    def training_loss(self, clean: torch.Tensor, labels: torch.Tensor,
                      condition_drop: float = 0.1):
        batch = clean.shape[0]
        time = torch.randint(len(self.betas), (batch,), device=clean.device)
        noise = torch.randn_like(clean)
        a_bar = self.alpha_bars[time, None]
        noisy = a_bar.sqrt() * clean + (1 - a_bar).sqrt() * noise
        chosen, _ = drop_conditions(labels, self.predictor.null_condition, condition_drop)
        return F.mse_loss(self.predictor(noisy, time, chosen), noise)

    @torch.no_grad()
    def sample(self, labels: torch.Tensor, scale: float = 3.0):
        x = torch.randn(labels.shape[0], self.predictor.input.in_features,
                        device=labels.device)
        for step in reversed(range(len(self.betas))):
            time = torch.full_like(labels, step)
            noise = guided_prediction(self.predictor, x, time, labels, scale)
            beta, alpha, a_bar = self.betas[step], self.alphas[step], self.alpha_bars[step]
            mean = (x - beta / (1 - a_bar).sqrt() * noise) / alpha.sqrt()
            if step:
                previous = self.alpha_bars[step - 1]
                posterior_variance = beta * (1 - previous) / (1 - a_bar)
                x = mean + posterior_variance.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    model = GuidedDiffusion(steps=8)
    labels = torch.tensor([0, 1, 2, 1])
    loss = model.training_loss(torch.randn(4, 4), labels)
    loss.backward()
    print("CFG loss/sample:", round(loss.item(), 4), tuple(model.sample(labels[:2]).shape))
