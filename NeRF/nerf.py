"""NeRF: positional encoding, density/color MLP and differentiable ray integration."""

import torch
from torch import nn
from torch.nn import functional as F


class FourierFeatures(nn.Module):
    def __init__(self, levels: int):
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(levels))
        self.output_dim = 3 * (1 + 2 * levels)

    def forward(self, xyz: torch.Tensor):
        angles = xyz[..., None, :] * self.frequencies[:, None] * torch.pi
        return torch.cat((xyz, angles.sin().flatten(-2), angles.cos().flatten(-2)), dim=-1)


class NeRF(nn.Module):
    def __init__(self, hidden: int = 64, xyz_levels: int = 6, dir_levels: int = 3):
        super().__init__()
        self.xyz_encoding = FourierFeatures(xyz_levels)
        self.dir_encoding = FourierFeatures(dir_levels)
        self.position_net = nn.Sequential(nn.Linear(self.xyz_encoding.output_dim, hidden), nn.ReLU(),
                                          nn.Linear(hidden, hidden), nn.ReLU(),
                                          nn.Linear(hidden, hidden), nn.ReLU())
        self.density = nn.Linear(hidden, 1)
        self.color = nn.Sequential(nn.Linear(hidden + self.dir_encoding.output_dim, hidden // 2),
                                   nn.ReLU(), nn.Linear(hidden // 2, 3), nn.Sigmoid())

    def forward(self, xyz: torch.Tensor, directions: torch.Tensor):
        """Matching leading shapes; return RGB [...,3] and nonnegative density [...]."""
        hidden = self.position_net(self.xyz_encoding(xyz))
        sigma = F.softplus(self.density(hidden)).squeeze(-1)
        rgb = self.color(torch.cat((hidden, self.dir_encoding(directions)), dim=-1))
        return rgb, sigma


def render_rays(model: NeRF, origins: torch.Tensor, directions: torch.Tensor,
                depths: torch.Tensor):
    """origins/directions [R,3], sorted depths [R,S] -> RGB/depth/opacity [R]."""
    if depths.ndim != 2 or depths.shape[1] < 2:
        raise ValueError("Each ray needs at least two ordered sample depths")
    points = origins[:, None, :] + depths[..., None] * directions[:, None, :]
    unit_dirs = F.normalize(directions, dim=-1)
    view_dirs = unit_dirs[:, None, :].expand_as(points)
    rgb, sigma = model(points, view_dirs)
    steps = depths[:, 1:] - depths[:, :-1]
    if torch.any(steps <= 0):
        raise ValueError("depths must increase along each ray")
    delta = torch.cat((steps, steps[:, -1:]), dim=-1) * directions.norm(dim=-1, keepdim=True)
    alpha = 1 - torch.exp(-sigma * delta)
    transmittance = torch.cumprod(torch.cat((torch.ones_like(alpha[:, :1]),
                                            1 - alpha + 1e-10), dim=-1), dim=-1)[:, :-1]
    weights = transmittance * alpha
    return (weights[..., None] * rgb).sum(dim=1), (weights * depths).sum(dim=1), weights.sum(dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = NeRF(hidden=32)
    origins = torch.zeros(4, 3)
    directions = torch.tensor([[0., 0., 1.]]).expand(4, 3)
    depths = torch.linspace(0.1, 1.0, 8).expand(4, 8)
    color, depth, opacity = render_rays(model, origins, directions, depths)
    color.mean().backward()
    print("NeRF RGB/depth/opacity:", tuple(color.shape), tuple(depth.shape), tuple(opacity.shape))
