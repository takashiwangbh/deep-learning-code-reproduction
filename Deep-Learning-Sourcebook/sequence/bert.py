"""BERT 式双向 Transformer 编码器与原论文的两项预训练输出。

MLM 预测被遮盖 token；NSP 用 [CLS] 位置预测两句话是否相接。
本文件只定义模型，训练数据中的 [MASK] 和标签需由调用方构造。
运行：python bert.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from transformer import MultiHeadAttention


class BERTBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid_key_mask: Tensor) -> Tensor:
        # 只有 PAD 不可见；没有下三角 mask，因此左右上下文都可读。
        x = self.attention_norm(x + self.dropout(self.attention(x, x, x, valid_key_mask)))
        return self.ffn_norm(x + self.dropout(self.ffn(x)))


class BERTEncoder(nn.Module):
    """token + 位置 + 句段嵌入，经双向自注意力得到 [B,T,D] 表示。"""

    def __init__(
        self,
        vocab_size: int,
        max_length: int = 128,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers 至少为 1")
        self.max_length = max_length
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.embedding_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            BERTBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        )

    def forward(
        self,
        token_ids: Tensor,
        token_type_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if token_ids.ndim != 2 or token_ids.size(1) == 0:
            raise ValueError("token_ids 应为非空的 [batch, time] 张量")
        batch, time = token_ids.shape
        if time > self.max_length:
            raise ValueError("输入长度超过 max_length")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(token_ids)
        if attention_mask is None:
            attention_mask = token_ids.ne(self.pad_id)
        if not bool(attention_mask.bool().any(dim=1).all()):
            raise ValueError("每个样本至少需要一个非 PAD token")
        positions = torch.arange(time, device=token_ids.device)
        x = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
            + self.segment_embedding(token_type_ids)
        )
        x = self.dropout(self.embedding_norm(x))
        valid_key_mask = attention_mask.bool()[:, None, None, :]
        for block in self.blocks:
            x = block(x, valid_key_mask)
        return x


class BERTForPretraining(nn.Module):
    """返回 MLM [B,T,V] 和 NSP [B,2] 的未归一化分数。"""

    def __init__(
        self,
        vocab_size: int,
        max_length: int = 128,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = BERTEncoder(
            vocab_size, max_length, d_model, num_heads, num_layers, d_ff, dropout, pad_id
        )
        self.mlm_transform = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.mlm_decoder = nn.Linear(d_model, vocab_size)
        self.mlm_decoder.weight = self.encoder.token_embedding.weight
        self.pooler = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.nsp_classifier = nn.Linear(d_model, 2)

    def forward(
        self,
        token_ids: Tensor,
        token_type_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(token_ids, token_type_ids, attention_mask)
        mlm_logits = self.mlm_decoder(self.mlm_transform(hidden))
        nsp_logits = self.nsp_classifier(self.pooler(hidden[:, 0]))
        return mlm_logits, nsp_logits


if __name__ == "__main__":
    model = BERTForPretraining(40, max_length=16, d_model=32, num_heads=4, d_ff=64, dropout=0)
    mlm, nsp = model(torch.tensor([[1, 3, 4, 0], [1, 5, 6, 7]]))
    print("MLM:", tuple(mlm.shape), "NSP:", tuple(nsp.shape))
