"""Small XLNet two-stream permutation language model (without Transformer-XL memory)."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PermutationAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError("Width must divide into heads")
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, allowed: torch.Tensor):
        batch, query_length, width = query.shape
        key_length = memory.shape[1]
        def split(tensor, length):
            return tensor.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        q = split(self.query(query), query_length)
        k = split(self.key(memory), key_length)
        v = split(self.value(memory), key_length)
        score = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_width)
        # A learned start token is always visible, even to the first prediction.
        score = score.masked_fill(~allowed[:, None], torch.finfo(score.dtype).min)
        weights = score.softmax(dim=-1)
        attended = (weights @ v).transpose(1, 2).reshape(batch, query_length, width)
        return self.output(attended)


class TwoStreamLayer(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.content_attention = PermutationAttention(width, heads)
        self.query_attention = PermutationAttention(width, heads)
        self.content_norm1, self.content_norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.query_norm1, self.query_norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.content_ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                        nn.Linear(4 * width, width))
        self.query_ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                      nn.Linear(4 * width, width))

    def forward(self, content, query, start, content_allowed, query_allowed):
        # Both streams read OLD content. Query never reads the token it predicts.
        memory = torch.cat((start.expand(content.shape[0], -1, -1), content), dim=1)
        new_content = self.content_norm1(
            content + self.content_attention(content, memory, content_allowed))
        new_content = self.content_norm2(new_content + self.content_ff(new_content))
        new_query = self.query_norm1(query + self.query_attention(query, memory, query_allowed))
        new_query = self.query_norm2(new_query + self.query_ff(new_query))
        return new_content, new_query


class XLNet(nn.Module):
    def __init__(self, vocab: int = 64, length: int = 16, width: int = 32,
                 heads: int = 4, layers: int = 2):
        super().__init__()
        self.words = nn.Embedding(vocab, width)
        self.positions = nn.Embedding(length, width)
        self.start = nn.Parameter(torch.zeros(1, 1, width))
        self.query_token = nn.Parameter(torch.zeros(1, 1, width))
        self.layers = nn.ModuleList(TwoStreamLayer(width, heads) for _ in range(layers))
        self.lm_head = nn.Linear(width, vocab)

    def forward(self, ids: torch.Tensor, order: torch.Tensor | None = None):
        """ids [B,T], order [B,T] lists positions in factorization order."""
        batch, time = ids.shape
        if time > self.positions.num_embeddings:
            raise ValueError("Sequence longer than configured maximum")
        if order is None:
            order = torch.rand(batch, time, device=ids.device).argsort(dim=-1)
        expected = torch.arange(time, device=ids.device).expand(batch, -1)
        if order.shape != ids.shape or not torch.equal(order.sort(dim=-1).values, expected):
            raise ValueError("Each row of order must permute positions 0..T-1")
        rank = order.argsort(dim=-1)
        content_allowed = rank[:, :, None] >= rank[:, None, :]
        query_allowed = rank[:, :, None] > rank[:, None, :]
        start_visible = torch.ones(batch, time, 1, dtype=torch.bool, device=ids.device)
        content_allowed = torch.cat((start_visible, content_allowed), dim=-1)
        query_allowed = torch.cat((start_visible, query_allowed), dim=-1)
        positions = self.positions(torch.arange(time, device=ids.device))[None]
        content = self.words(ids) + positions
        query = self.query_token.expand(batch, time, -1) + positions
        for layer in self.layers:
            content, query = layer(content, query, self.start, content_allowed, query_allowed)
        return self.lm_head(query), order

    def training_loss(self, ids: torch.Tensor, order: torch.Tensor | None = None):
        logits, chosen_order = self(ids, order)
        return F.cross_entropy(logits.flatten(0, 1), ids.flatten()), chosen_order


if __name__ == "__main__":
    torch.manual_seed(0)
    model = XLNet(vocab=32, length=6, width=16)
    ids = torch.randint(0, 32, (2, 5))
    loss, order = model.training_loss(ids)
    loss.backward()
    print("XLNet permutation/loss:", order.tolist(), round(loss.item(), 4))
