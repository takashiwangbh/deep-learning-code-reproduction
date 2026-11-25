"""LSTM gates and cell state implemented without nn.LSTM."""

import torch
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_map = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_map = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h, c = state
        i, f, g, o = (self.input_map(x) + self.hidden_map(h)).chunk(4, dim=-1)
        i, f, o = i.sigmoid(), f.sigmoid(), o.sigmoid()
        c = f * c + i * g.tanh()
        h = o * c.tanh()
        return h, c


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, state=None):
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError("x must be [B,T,D] with T>0")
        batch = x.shape[0]
        if state is None:
            state = (x.new_zeros(batch, self.hidden_size), x.new_zeros(batch, self.hidden_size))
        outputs = []
        for step in x.unbind(dim=1):
            state = self.cell(step, state)
            outputs.append(state[0])
        return torch.stack(outputs, dim=1), state


if __name__ == "__main__":
    model = LSTM(4, 8)
    sequence, (h, c) = model(torch.randn(2, 5, 4))
    sequence.sum().backward()
    print("outputs:", tuple(sequence.shape), "cell:", tuple(c.shape))
