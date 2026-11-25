"""Bidirectional LSTM with explicit gates in both temporal directions."""

import torch
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_map = nn.Linear(input_size, hidden_size * 4)
        self.hidden_map = nn.Linear(hidden_size, hidden_size * 4, bias=False)

    def forward(self, x, h, c):
        i, f, g, o = (self.input_map(x) + self.hidden_map(h)).chunk(4, dim=-1)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        return o.sigmoid() * c.tanh(), c


class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.forward_cell = LSTMCell(input_size, hidden_size)
        self.backward_cell = LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def _scan(self, x, cell, reverse=False):
        h = x.new_zeros(x.shape[0], self.hidden_size)
        c = torch.zeros_like(h)
        sequence = list(x.unbind(1))
        if reverse:
            sequence.reverse()
        outputs = []
        for step in sequence:
            h, c = cell(step, h, c)
            outputs.append(h)
        if reverse:
            outputs.reverse()
        return torch.stack(outputs, dim=1)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError("x must be [B,T,D] with T>0")
        return torch.cat([self._scan(x, self.forward_cell), self._scan(x, self.backward_cell, True)], dim=-1)


if __name__ == "__main__":
    model = BiLSTM(4, 8)
    y = model(torch.randn(2, 5, 4))
    y.sum().backward()
    print("BiLSTM states:", tuple(y.shape))
