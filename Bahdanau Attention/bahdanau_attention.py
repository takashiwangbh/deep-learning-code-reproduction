"""Additive alignment with explicit recurrent encoder and decoder."""

import torch
from torch import nn


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.x_gate = nn.Linear(input_size, 2 * hidden_size)
        self.h_gate = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.x_new = nn.Linear(input_size, hidden_size)
        self.h_new = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        xr, xz = self.x_gate(x).chunk(2, -1)
        hr, hz = self.h_gate(h).chunk(2, -1)
        r, z = (xr + hr).sigmoid(), (xz + hz).sigmoid()
        candidate = torch.tanh(self.x_new(x) + r * self.h_new(h))
        return z * h + (1 - z) * candidate


class AdditiveAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.score = nn.Linear(width, 1, bias=False)

    def forward(self, query: torch.Tensor, values: torch.Tensor, valid: torch.Tensor | None = None):
        scores = self.score(torch.tanh(self.query(query).unsqueeze(1) + self.key(values))).squeeze(-1)
        if valid is not None:
            if valid.shape != scores.shape or not torch.all(valid.any(dim=-1)):
                raise ValueError("valid must be [B,S] with at least one unmasked source token per item")
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), values).squeeze(1), weights


class BahdanauSeq2Seq(nn.Module):
    def __init__(self, source_vocab: int, target_vocab: int, width: int = 32):
        super().__init__()
        self.source_embedding = nn.Embedding(source_vocab, width)
        self.target_embedding = nn.Embedding(target_vocab, width)
        self.encoder = GRUCell(width, width)
        self.attention = AdditiveAttention(width)
        self.decoder = GRUCell(2 * width, width)
        self.head = nn.Linear(width, target_vocab)
        self.width = width

    def forward(self, source: torch.Tensor, target_inputs: torch.Tensor, source_valid=None):
        if source_valid is not None and source_valid.shape != source.shape:
            raise ValueError("source_valid must be [B,S]")
        h = self.source_embedding.weight.new_zeros(source.shape[0], self.width)
        encoded = []
        for step, token in enumerate(self.source_embedding(source).unbind(1)):
            proposed = self.encoder(token, h)
            h = proposed if source_valid is None else torch.where(source_valid[:, step, None], proposed, h)
            encoded.append(h)
        memory = torch.stack(encoded, 1)
        logits, alignments = [], []
        for token in self.target_embedding(target_inputs).unbind(1):
            context, weights = self.attention(h, memory, source_valid)
            h = self.decoder(torch.cat([token, context], dim=-1), h)
            logits.append(self.head(h))
            alignments.append(weights)
        return torch.stack(logits, 1), torch.stack(alignments, 1)


if __name__ == "__main__":
    model = BahdanauSeq2Seq(20, 25, 8)
    logits, attention = model(torch.randint(0, 20, (2, 5)), torch.randint(0, 25, (2, 4)))
    logits.sum().backward()
    print("logits:", tuple(logits.shape), "alignment:", tuple(attention.shape))
