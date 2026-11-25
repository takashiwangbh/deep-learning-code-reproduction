"""SASRec: left-to-right item self-attention for next-item recommendation."""

import torch
from torch import nn


class SASBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, 4 * width), nn.ReLU(),
                                         nn.Linear(4 * width, width))

    def forward(self, x: torch.Tensor, causal: torch.Tensor, padding: torch.Tensor):
        attended, _ = self.attention(x, x, x, attn_mask=causal,
                                     key_padding_mask=padding, need_weights=False)
        x = self.norm1(x + attended)
        return self.norm2(x + self.feedforward(x))


class SASRec(nn.Module):
    def __init__(self, items: int, length: int = 20, width: int = 32,
                 heads: int = 4, layers: int = 2):
        super().__init__()
        self.item_embedding = nn.Embedding(items + 1, width, padding_idx=0)
        self.position_embedding = nn.Embedding(length, width)
        self.blocks = nn.ModuleList(SASBlock(width, heads) for _ in range(layers))

    def forward(self, history: torch.Tensor):
        """Right-padded item IDs [B,T] -> next-item logits [B,T,items+1]."""
        batch, length = history.shape
        if length > self.position_embedding.num_embeddings or torch.any(history[:, 0] == 0):
            raise ValueError("Sequences must begin with an item and fit configured length")
        padding = history.eq(0)
        position = torch.arange(length, device=history.device)
        x = self.item_embedding(history) + self.position_embedding(position)[None]
        causal = torch.ones(length, length, device=history.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            x = block(x, causal, padding)
            x = x.masked_fill(padding[..., None], 0)
        return x @ self.item_embedding.weight.T


if __name__ == "__main__":
    torch.manual_seed(0)
    model = SASRec(items=20, length=5, width=16)
    history = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]])
    logits = model(history)
    logits[:, :3].sum().backward()
    print("SASRec next-item logits:", tuple(logits.shape))
