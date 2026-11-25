"""DINO: teacher/student ViTs, cross-view distillation, centering and EMA."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class TinyVisionTransformer(nn.Module):
    def __init__(self, image_size: int = 32, patch: int = 8, dim: int = 64,
                 prototypes: int = 128):
        super().__init__()
        if image_size % patch:
            raise ValueError("image size must be divisible by patch size")
        self.patch_embed = nn.Conv2d(3, dim, patch, stride=patch)
        patches = (image_size // patch) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.randn(1, patches + 1, dim) * 0.02)
        block = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=2 * dim,
                                           batch_first=True, dropout=0)
        self.blocks = nn.TransformerEncoder(block, num_layers=2)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, prototypes))

    def forward(self, images: torch.Tensor):
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        if patches.shape[1] + 1 != self.position.shape[1]:
            raise ValueError("This teaching model expects its configured image size")
        tokens = torch.cat((self.cls_token.expand(images.shape[0], -1, -1), patches), dim=1)
        return self.head(self.blocks(tokens + self.position)[:, 0])


class DINO(nn.Module):
    def __init__(self, dim: int = 64, prototypes: int = 128, momentum: float = 0.996,
                 student_temperature: float = 0.1, teacher_temperature: float = 0.04):
        super().__init__()
        self.student = TinyVisionTransformer(dim=dim, prototypes=prototypes)
        self.teacher = deepcopy(self.student)
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.register_buffer("center", torch.zeros(1, prototypes))
        self.momentum = momentum
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature

    @torch.no_grad()
    def update_teacher(self):
        for target, online in zip(self.teacher.parameters(), self.student.parameters()):
            target.lerp_(online, 1 - self.momentum)

    def forward(self, view_a: torch.Tensor, view_b: torch.Tensor):
        student_a, student_b = self.student(view_a), self.student(view_b)
        with torch.no_grad():
            teacher_a, teacher_b = self.teacher(view_a), self.teacher(view_b)
            prob_a = ((teacher_a - self.center) / self.teacher_temperature).softmax(dim=-1)
            prob_b = ((teacher_b - self.center) / self.teacher_temperature).softmax(dim=-1)
        log_a = F.log_softmax(student_a / self.student_temperature, dim=-1)
        log_b = F.log_softmax(student_b / self.student_temperature, dim=-1)
        loss = -0.5 * ((prob_b * log_a).sum(-1).mean() + (prob_a * log_b).sum(-1).mean())
        with torch.no_grad():
            self.center.lerp_(torch.cat((teacher_a, teacher_b)).mean(dim=0, keepdim=True), 0.1)
        return loss


if __name__ == "__main__":
    torch.manual_seed(0)
    model = DINO(dim=32, prototypes=64)
    optimizer = torch.optim.Adam(model.student.parameters(), lr=1e-3)
    images = torch.randn(2, 3, 32, 32)
    loss = model(images, images + 0.03 * torch.randn_like(images))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.update_teacher()
    print("DINO cross-view loss:", round(loss.item(), 4))
