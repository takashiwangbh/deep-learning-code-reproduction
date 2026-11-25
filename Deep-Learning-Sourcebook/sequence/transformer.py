"""原始 Transformer 的编码器、解码器及多头注意力。

没有调用 nn.Transformer / nn.MultiheadAttention；注意力矩阵在这里显式计算。
运行：python transformer.py
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def causal_mask(length: int, device: torch.device) -> Tensor:
    """True 表示可以读取；第 t 个位置不能看未来的 token。"""
    return torch.ones(length, length, device=device, dtype=torch.bool).tril()


class MultiHeadAttention(nn.Module):
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V。"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model 必须能被 num_heads 整除")
        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, time, _ = x.shape
        return x.reshape(batch, time, self.num_heads, self.head_size).transpose(1, 2)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        # 分头后的 Q/K/V: [B,H,T,D]；分数矩阵: [B,H,T_q,T_k]。
        q = self._split_heads(self.query(query))
        k = self._split_heads(self.key(key))
        v = self._split_heads(self.value(value))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        if mask is not None:
            # 若整行全是 PAD，softmax(finfo.min, ...) 会变成均匀分布并读到未来。
            # 再把无效位置显式清零，使这一行没有上下文信息。
            weights = weights.masked_fill(~mask, 0.0)
        weights = self.dropout(weights)
        context = weights @ v
        context = context.transpose(1, 2).contiguous().reshape(query.size(0), query.size(1), -1)
        return self.output(context)


class PositionalEncoding(nn.Module):
    """固定正弦位置编码：偶数维 sin，奇数维 cos。"""

    def __init__(self, d_model: int, max_length: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        angles = position * frequency
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].size(1)])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) > self.encoding.size(1):
            raise ValueError("输入长度超过 max_length")
        return x + self.encoding[:, : x.size(1)].to(dtype=x.dtype)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, source_mask: Tensor) -> Tensor:
        # 论文的 Add & Norm：子层输出和输入相加，再归一化。
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x, source_mask)))
        return self.norm2(x + self.dropout(self.feed_forward(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, memory: Tensor, target_mask: Tensor, source_mask: Tensor) -> Tensor:
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x, target_mask)))
        x = self.norm2(x + self.dropout(self.cross_attention(x, memory, memory, source_mask)))
        return self.norm3(x + self.dropout(self.feed_forward(x)))


class TransformerSeq2Seq(nn.Module):
    """原始 encoder-decoder Transformer；训练目标为下一个目标 token。

    source_ids [B,S]，target_input_ids [B,T]；返回 [B,T,V_tgt]。
    """

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        max_length: int = 512,
        pad_id: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers 至少为 1")
        self.pad_id = pad_id
        self.scale = math.sqrt(d_model)
        self.source_embedding = nn.Embedding(source_vocab_size, d_model)
        self.target_embedding = nn.Embedding(target_vocab_size, d_model)
        self.position = PositionalEncoding(d_model, max_length)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        )
        self.decoder = nn.ModuleList(
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        )
        self.output = nn.Linear(d_model, target_vocab_size)

    def forward(self, source_ids: Tensor, target_input_ids: Tensor) -> Tensor:
        if (
            source_ids.ndim != 2
            or target_input_ids.ndim != 2
            or source_ids.size(1) == 0
            or target_input_ids.size(1) == 0
        ):
            raise ValueError("输入应为非空的 [batch, time] token ID 张量")
        if not bool(source_ids.ne(self.pad_id).any(dim=1).all()):
            raise ValueError("每个源序列至少需要一个非 PAD token")
        if not bool(target_input_ids.ne(self.pad_id).any(dim=1).all()):
            raise ValueError("每个目标序列至少需要一个非 PAD token")
        # key padding mask: [B,1,1,S]；解码器 mask: [B,1,T,T]。
        source_mask = source_ids.ne(self.pad_id)[:, None, None, :]
        target_valid = target_input_ids.ne(self.pad_id)[:, None, None, :]
        target_mask = target_valid & causal_mask(target_input_ids.size(1), target_input_ids.device)
        memory = self.dropout(self.position(self.source_embedding(source_ids) * self.scale))
        for layer in self.encoder:
            memory = layer(memory, source_mask)
        x = self.dropout(self.position(self.target_embedding(target_input_ids) * self.scale))
        for layer in self.decoder:
            x = layer(x, memory, target_mask, source_mask)
        return self.output(x)


if __name__ == "__main__":
    model = TransformerSeq2Seq(30, 40, d_model=32, num_heads=4, d_ff=64, dropout=0)
    logits = model(torch.tensor([[2, 3, 4, 0]]), torch.tensor([[1, 7, 8]]))
    print("logits:", tuple(logits.shape))
