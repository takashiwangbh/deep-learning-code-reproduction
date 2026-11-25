"""An Elman recurrent network written one time step at a time."""

import torch
from torch import nn


class VanillaRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_map = nn.Linear(input_size, hidden_size)
        self.hidden_map = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        if x.ndim != 3:
            raise ValueError("x must be [B,T,D]")
        batch, steps, _ = x.shape
        h = x.new_zeros(batch, self.hidden_size) if state is None else state
        outputs = []
        for t in range(steps):
            h = torch.tanh(self.input_map(x[:, t]) + self.hidden_map(h))
            outputs.append(h)
        return torch.stack(outputs, dim=1), h


if __name__ == "__main__":
    model = VanillaRNN(4, 8)
    output, final_state = model(torch.randn(2, 5, 4))
    output.sum().backward()
    print("states:", tuple(output.shape), "last:", tuple(final_state.shape))
