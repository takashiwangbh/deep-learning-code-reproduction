"""Encoder-decoder with explicit GRU gates and teacher-forced decoding; no attention."""

import torch
from torch import nn


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_gates = nn.Linear(input_size, 2 * hidden_size)
        self.hidden_gates = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.input_candidate = nn.Linear(input_size, hidden_size)
        self.hidden_candidate = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        xr, xz = self.input_gates(x).chunk(2, -1)
        hr, hz = self.hidden_gates(h).chunk(2, -1)
        r, z = (xr + hr).sigmoid(), (xz + hz).sigmoid()
        n = torch.tanh(self.input_candidate(x) + r * self.hidden_candidate(h))
        return (1 - z) * n + z * h


class Seq2Seq(nn.Module):
    def __init__(self, source_vocab: int, target_vocab: int, width: int = 32):
        super().__init__()
        self.source_embedding = nn.Embedding(source_vocab, width)
        self.target_embedding = nn.Embedding(target_vocab, width)
        self.encoder = GRUCell(width, width)
        self.decoder = GRUCell(width, width)
        self.output = nn.Linear(width, target_vocab)
        self.width = width

    def encode(self, source_tokens):
        h = self.source_embedding.weight.new_zeros(source_tokens.shape[0], self.width)
        for token in self.source_embedding(source_tokens).unbind(1):
            h = self.encoder(token, h)
        return h

    def forward(self, source_tokens: torch.Tensor, target_inputs: torch.Tensor):
        """target_inputs begins with BOS; logits[t] predict the following token."""
        h = self.encode(source_tokens)
        logits = []
        for token in self.target_embedding(target_inputs).unbind(1):
            h = self.decoder(token, h)
            logits.append(self.output(h))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def generate(self, source_tokens: torch.Tensor, bos_token: int, steps: int):
        h = self.encode(source_tokens)
        current = torch.full((source_tokens.shape[0],), bos_token, dtype=torch.long, device=source_tokens.device)
        tokens = []
        for _ in range(steps):
            h = self.decoder(self.target_embedding(current), h)
            current = self.output(h).argmax(dim=-1)
            tokens.append(current)
        return torch.stack(tokens, dim=1)


if __name__ == "__main__":
    model = Seq2Seq(20, 25, 8)
    y = model(torch.randint(0, 20, (2, 5)), torch.randint(0, 25, (2, 4)))
    y.sum().backward()
    print("decoder logits:", tuple(y.shape), "generated:", tuple(model.generate(torch.randint(0, 20, (2, 5)), 1, 3).shape))
