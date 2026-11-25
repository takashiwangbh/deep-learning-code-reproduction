"""GRU reset, update and candidate gates implemented without nn.GRU."""

import torch
from torch import nn


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_gates = nn.Linear(input_size, 2 * hidden_size)
        self.hidden_gates = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.input_new = nn.Linear(input_size, hidden_size)
        self.hidden_new = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x_r, x_z = self.input_gates(x).chunk(2, -1)
        h_r, h_z = self.hidden_gates(h).chunk(2, -1)
        reset = torch.sigmoid(x_r + h_r)
        update = torch.sigmoid(x_z + h_z)
        candidate = torch.tanh(self.input_new(x) + reset * self.hidden_new(h))
        return (1 - update) * candidate + update * h


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError("x must be [B,T,D] with T>0")
        h = x.new_zeros(x.shape[0], self.hidden_size) if state is None else state
        outputs = []
        for step in x.unbind(dim=1):
            h = self.cell(step, h)
            outputs.append(h)
        return torch.stack(outputs, dim=1), h


if __name__ == "__main__":
    model = GRU(4, 8)
    sequence, h = model(torch.randn(2, 5, 4))
    sequence.sum().backward()
    print("outputs:", tuple(sequence.shape), "last:", tuple(h.shape))
