"""纯 PyTorch 的 Mamba-1 风格选择性状态空间块，显式展示逐时间步扫描。

这是可学习的教学实现，复杂度在序列长度上为 O(T)，但 Python 循环不具备论文 CUDA
并行扫描的吞吐量；也未实现原版全部初始化和融合算子。
"""

import torch
import torch.nn.functional as F
from torch import nn


class MambaBlock(nn.Module):
    def __init__(self, dim: int, state_dim: int = 16, expand: int = 2,
                 conv_kernel: int = 4, dt_rank: int | None = None):
        super().__init__()
        self.inner_dim = dim * expand
        self.state_dim = state_dim
        dt_rank = dt_rank or max(1, dim // 16)
        self.in_proj = nn.Linear(dim, 2 * self.inner_dim, bias=False)
        self.conv = nn.Conv1d(self.inner_dim, self.inner_dim, conv_kernel,
                              padding=conv_kernel - 1, groups=self.inner_dim)
        self.x_proj = nn.Linear(self.inner_dim, dt_rank + 2 * state_dim, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.inner_dim)
        # A=-exp(A_log) 始终为负，因此离散衰减 exp(ΔA) 在 (0,1) 内。
        initial_a = torch.arange(1, state_dim + 1).float().log()
        self.A_log = nn.Parameter(initial_a.repeat(self.inner_dim, 1))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        # Conv1d 的右端多出的 k-1 个值被裁掉：第 t 个输出只看 0..t。
        u = self.conv(u.transpose(1, 2))[..., :length].transpose(1, 2)
        u = F.silu(u)
        dt_bc = self.x_proj(u)
        dt_raw, B, C = torch.split(
            dt_bc, [self.dt_proj.in_features, self.state_dim, self.state_dim], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_raw))            # [B, T, inner]
        A = -torch.exp(self.A_log)                        # [inner, state]
        state = x.new_zeros(batch, self.inner_dim, self.state_dim)
        outputs = []
        for t in range(length):
            # h_t = exp(Δ_t A) ⊙ h_(t-1) + Δ_t B_t ⊙ u_t
            delta_a = torch.exp(dt[:, t, :, None] * A[None])
            delta_b_u = dt[:, t, :, None] * B[:, t, None, :] * u[:, t, :, None]
            state = delta_a * state + delta_b_u
            y_t = (state * C[:, t, None, :]).sum(dim=-1) + self.D * u[:, t]
            outputs.append(y_t * F.silu(gate[:, t]))
        return self.out_proj(torch.stack(outputs, dim=1))


class MambaLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 64, depth: int = 2,
                 state_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(depth))
        self.blocks = nn.ModuleList(MambaBlock(dim, state_dim) for _ in range(depth))
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))
        return self.head(self.final_norm(x))


if __name__ == "__main__":
    model = MambaLanguageModel(vocab_size=32, dim=16, state_dim=4)
    tokens = torch.randint(0, 32, (2, 8))
    logits = model(tokens)
    F.cross_entropy(logits[:, :-1].reshape(-1, 32), tokens[:, 1:].reshape(-1)).backward()
    print("logits:", tuple(logits.shape))
