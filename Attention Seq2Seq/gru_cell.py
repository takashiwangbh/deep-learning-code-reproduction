"""Seq2Seq 使用的 GRU 单元：把门控源码留在本项目内。"""

import torch
from torch import Tensor, nn


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_to_gates = nn.Linear(input_size, 3 * hidden_size)
        self.hidden_to_gates = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def forward(self, x: Tensor, hidden: Tensor) -> Tensor:
        x_r, x_z, x_n = self.input_to_gates(x).chunk(3, dim=-1)
        h_r, h_z, h_n = self.hidden_to_gates(hidden).chunk(3, dim=-1)
        reset = torch.sigmoid(x_r + h_r)
        update = torch.sigmoid(x_z + h_z)
        candidate = torch.tanh(x_n + reset * h_n)
        return update * hidden + (1 - update) * candidate
