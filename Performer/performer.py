"""Performer 核心：正随机特征近似 softmax 核并用递推做因果注意力。"""
import math
import torch
from torch import nn


class FavorAttention(nn.Module):
    def __init__(self, dim=64, heads=4, features=32):
        super().__init__()
        assert dim % heads == 0 and features > 0
        self.heads, self.width, self.features = heads, dim // heads, features
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.register_buffer("omega", torch.randn(heads, self.width, features))

    def _log_feature(self, x):
        x = x * self.width ** -0.25
        return torch.einsum("bhtd,hdm->bhtm", x, self.omega) - x.square().sum(-1, keepdim=True) / 2

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, self.width).permute(2, 0, 3, 1, 4)
        q_log, k_log = self._log_feature(q), self._log_feature(k)
        # query 的缩放只依赖自身，分子/分母会相互抵消。
        q = torch.exp(q_log - q_log.amax(-1, keepdim=True)) / math.sqrt(self.features)
        kv_state = torch.zeros(b, self.heads, self.features, self.width, device=x.device, dtype=x.dtype)
        k_state = torch.zeros(b, self.heads, self.features, device=x.device, dtype=x.dtype)
        key_shift = torch.full((b, self.heads, 1), -torch.inf, device=x.device, dtype=x.dtype)
        outputs = []
        for i in range(t):
            # 历史 key 的公共缩放只看当前及过去。最大值变大时同步重标定旧状态。
            new_shift = torch.maximum(key_shift, k_log[:, :, i].amax(-1, keepdim=True))
            rescale = torch.exp(key_shift - new_shift)
            current_key = torch.exp(k_log[:, :, i] - new_shift) / math.sqrt(self.features)
            kv_state = kv_state * rescale[..., None] + current_key[..., None] * v[:, :, i, None, :]
            k_state = k_state * rescale + current_key
            key_shift = new_shift
            numerator = torch.einsum("bhm,bhmd->bhd", q[:, :, i], kv_state)
            denominator = (q[:, :, i] * k_state).sum(-1, keepdim=True).clamp_min(1e-8)
            outputs.append(numerator / denominator)
        return self.out(torch.stack(outputs, dim=2).transpose(1, 2).reshape(b, t, d))


class Performer(nn.Module):
    def __init__(self, vocab=128, dim=64, heads=4, features=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.attn = FavorAttention(dim, heads, features)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids):
        x = self.embed(ids)
        return self.head(self.norm(x + self.attn(x)))


if __name__ == "__main__":
    model = Performer(vocab=32, dim=32, features=16)
    logits = model(torch.randint(0, 32, (2, 5)))
    logits.sum().backward()
    print("Performer", tuple(logits.shape))
