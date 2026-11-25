"""Gated DeltaNet-2 (2026) 的独立擦除/写入递推，纯 PyTorch 教学版。

论文：https://arxiv.org/abs/2605.22791
保留 Eq. 9/10 的逐通道递推、输出 RMSNorm/SiLU 门控；为便于读懂，省略
分块并行 CUDA、短因果卷积、分组 value head 与混合模型的滑动窗口注意力。
"""

import torch
import torch.nn.functional as F
from torch import nn

from modern_decoder import RMSNorm, SwiGLU


def gated_delta2_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      alpha: torch.Tensor, erase: torch.Tensor,
                      write: torch.Tensor) -> torch.Tensor:
    """六个输入均为 [B,T,H,D]，输出为 [B,T,H,D]。

    S̄ₜ=Diag(αₜ)Sₜ₋₁；rₜ=S̄ₜᵀ(bₜ⊙kₜ)；
    Sₜ=S̄ₜ+kₜ(wₜ⊙vₜ−rₜ)ᵀ；oₜ=Sₜᵀqₜ。
    state 的行是 key 通道、列是 value 通道。
    """
    if not all(tensor.shape == q.shape for tensor in (k, v, alpha, erase, write)):
        raise ValueError("q, k, v, alpha, erase and write must have the same shape")
    batch, length, heads, dim = q.shape
    output_dtype = q.dtype
    # 递推状态始终至少用 fp32 累积，避免长序列的 fp16/bf16 误差放大。
    compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v, alpha, erase, write = (
        tensor.to(compute_dtype) for tensor in (q, k, v, alpha, erase, write)
    )
    state = q.new_zeros(batch, heads, dim, dim)
    outputs = []
    for t in range(length):
        key = k[:, t]
        decayed = alpha[:, t, :, :, None] * state
        old_value = ((erase[:, t] * key)[..., None] * decayed).sum(dim=-2)
        new_value = write[:, t] * v[:, t]
        state = decayed + key[..., None] * (new_value - old_value)[..., None, :]
        outputs.append((q[:, t, ..., None] * state).sum(dim=-2))
    return torch.stack(outputs, dim=1).to(dtype=output_dtype)


class HeadRMSNorm(nn.Module):
    def __init__(self, heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(heads, head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        wide = x.to(compute_dtype)
        result = wide * torch.rsqrt(wide.square().mean(dim=-1, keepdim=True) + self.eps)
        return (result * self.weight.to(compute_dtype)).to(x.dtype)


class GatedDelta2Mixer(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads or dim // heads < 2:
            raise ValueError("dim must be divisible by heads and head_dim >= 2")
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.decay_proj = nn.Linear(dim, dim)
        self.erase_proj = nn.Linear(dim, dim)
        self.write_proj = nn.Linear(dim, dim)
        # 论文附录 C.1：a 每个 head 一个值，广播到该 head 的 key 通道。
        self.decay_log_scale = nn.Parameter(torch.zeros(heads, 1))
        self.output_norm = HeadRMSNorm(heads, self.head_dim)
        self.output_gate = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        shape = (batch, length, self.heads, self.head_dim)
        q, k, v = (part.reshape(shape) for part in self.qkv(x).chunk(3, dim=-1))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        # log α = -exp(a) · softplus(W_f x + bias)；每个 key 通道可独立衰减。
        raw_decay = self.decay_proj(x).reshape(shape)
        alpha = torch.exp(-self.decay_log_scale.float().exp() * F.softplus(raw_decay.float()))
        erase = self.erase_proj(x).reshape(shape).sigmoid()
        write = self.write_proj(x).reshape(shape).sigmoid()
        output = gated_delta2_scan(q, k, v, alpha, erase, write)
        gate = F.silu(self.output_gate(x)).reshape(shape)
        return self.out_proj((self.output_norm(output) * gate).reshape(batch, length, dim))


class GatedDelta2Block(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.mixer = GatedDelta2Mixer(dim, heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class GatedDelta2LanguageModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 64, depth: int = 2,
                 heads: int = 4, hidden: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(GatedDelta2Block(dim, heads, hidden) for _ in range(depth))
        self.final_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))


if __name__ == "__main__":
    model = GatedDelta2LanguageModel(32, dim=16, depth=2, heads=4, hidden=32)
    tokens = torch.randint(0, 32, (2, 6))
    logits = model(tokens)
    F.cross_entropy(logits[:, :-1].reshape(-1, 32), tokens[:, 1:].reshape(-1)).backward()
    print("Gated DeltaNet-2 logits:", tuple(logits.shape))
