"""教学版 T5：文本到文本、相对位置桶、RMSNorm 与编码器-解码器。"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def relative_bucket(relative, bidirectional, buckets=32, max_distance=128):
    """T5 的线性小距离 + 对数大距离分桶。relative = key_pos - query_pos。"""
    n = -relative
    result = torch.zeros_like(n)
    if bidirectional:
        half = buckets // 2
        result = (n < 0).long() * half
        n = n.abs()
        buckets = half
    else:
        n = n.clamp_min(0)
    exact = buckets // 2
    large = exact + (
        torch.log(n.float().clamp_min(1) / exact)
        / math.log(max_distance / exact) * (buckets - exact)
    ).long()
    return result + torch.where(n < exact, n, large.clamp(max=buckets - 1))


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, dim, heads, relative=False):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.width = heads, dim // heads
        self.q, self.k, self.v, self.out = [nn.Linear(dim, dim, bias=False) for _ in range(4)]
        self.bias = nn.Embedding(32, heads) if relative else None

    def forward(self, x, context=None, mask=None, causal=False):
        context = x if context is None else context
        b, t, d = x.shape
        s = context.size(1)
        split = lambda z: z.reshape(b, -1, self.heads, self.width).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(context)), split(self.v(context))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.width)
        if self.bias is not None:
            rel = torch.arange(s, device=x.device)[None, :] - torch.arange(t, device=x.device)[:, None]
            buckets = relative_bucket(rel, bidirectional=not causal)
            scores = scores + self.bias(buckets).permute(2, 0, 1)[None]
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        if causal:
            allowed = torch.ones(t, s, device=x.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weight = scores.softmax(-1)
        if mask is not None:
            weight = weight.masked_fill(~mask[:, None, None, :], 0)
        return self.out((weight @ v).transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, dim, heads, decoder=False):
        super().__init__()
        self.decoder = decoder
        self.norm1, self.self_attn = RMSNorm(dim), Attention(dim, heads, relative=True)
        if decoder:
            self.norm_cross, self.cross_attn = RMSNorm(dim), Attention(dim, heads)
        self.norm_ff = RMSNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim, bias=False), nn.ReLU(), nn.Linear(4 * dim, dim, bias=False))

    def forward(self, x, own_mask, memory=None, memory_mask=None):
        x = x + self.self_attn(self.norm1(x), mask=own_mask, causal=self.decoder)
        if self.decoder:
            x = x + self.cross_attn(self.norm_cross(x), memory, mask=memory_mask)
        return x + self.ff(self.norm_ff(x))


class T5(nn.Module):
    def __init__(self, vocab=128, dim=64, heads=4, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.encoder = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.decoder = nn.ModuleList(Block(dim, heads, decoder=True) for _ in range(layers))
        self.enc_norm, self.dec_norm = RMSNorm(dim), RMSNorm(dim)
        self.output = nn.Linear(dim, vocab, bias=False)
        self.output.weight = self.embedding.weight

    def forward(self, source_ids, decoder_ids):
        source_mask, target_mask = source_ids.ne(0), decoder_ids.ne(0)
        if not source_mask.any(-1).all() or not target_mask.any(-1).all():
            raise ValueError("每条输入至少有一个非 PAD token")
        memory = self.embedding(source_ids)
        for layer in self.encoder:
            memory = layer(memory, source_mask)
        memory = self.enc_norm(memory)
        x = self.embedding(decoder_ids)
        for layer in self.decoder:
            x = layer(x, target_mask, memory, source_mask)
        return self.output(self.dec_norm(x) / math.sqrt(self.embedding.embedding_dim))


if __name__ == "__main__":
    model = T5(vocab=32, dim=32, layers=1)
    logits = model(torch.tensor([[3, 4, 5, 0]]), torch.tensor([[1, 8, 9]]))
    loss = F.cross_entropy(logits.reshape(-1, 32), torch.tensor([8, 9, 2]))
    loss.backward()
    print("T5 logits", tuple(logits.shape), "loss", round(loss.item(), 4))
