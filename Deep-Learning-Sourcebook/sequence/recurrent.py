"""手写 RNN / LSTM / GRU 单元及一个最小语言模型。

只借助 PyTorch 的张量运算与线性层；时间展开和门控方程都在这里。
运行：python recurrent.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RNNCell(nn.Module):
    """h_t = tanh(W_x x_t + W_h h_(t-1) + b)。"""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: Tensor, hidden: Tensor) -> Tensor:
        return torch.tanh(self.input_to_hidden(x) + self.hidden_to_hidden(hidden))


class LSTMCell(nn.Module):
    """显式计算输入门 i、遗忘门 f、候选记忆 g、输出门 o。"""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_to_gates = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_to_gates = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        hidden, cell = state
        i, f, g, o = (self.input_to_gates(x) + self.hidden_to_gates(hidden)).chunk(4, dim=-1)
        # c_t = f_t * c_(t-1) + i_t * g_t；长期信息走加法路径。
        cell = torch.sigmoid(f) * cell + torch.sigmoid(i) * torch.tanh(g)
        hidden = torch.sigmoid(o) * torch.tanh(cell)
        return hidden, cell


class GRUCell(nn.Module):
    """显式计算重置门 r、更新门 z 与候选状态 n。"""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_to_gates = nn.Linear(input_size, 3 * hidden_size)
        self.hidden_to_gates = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def forward(self, x: Tensor, hidden: Tensor) -> Tensor:
        x_r, x_z, x_n = self.input_to_gates(x).chunk(3, dim=-1)
        h_r, h_z, h_n = self.hidden_to_gates(hidden).chunk(3, dim=-1)
        reset = torch.sigmoid(x_r + h_r)
        update = torch.sigmoid(x_z + h_z)
        candidate = torch.tanh(x_n + reset * h_n)
        # z 接近 1 时保留旧状态；接近 0 时采纳新候选。
        return update * hidden + (1 - update) * candidate


class RecurrentLanguageModel(nn.Module):
    """单层循环语言模型：token -> embedding -> 手写循环单元 -> logits。

    输入 tokens: [batch, time]；输出 logits: [batch, time, vocab_size]。
    训练时将 tokens[:, :-1] 作为输入，tokens[:, 1:] 作为标签。
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 64,
        hidden_size: int = 128,
        cell_type: str = "lstm",
    ) -> None:
        super().__init__()
        if cell_type not in {"rnn", "lstm", "gru"}:
            raise ValueError("cell_type 必须是 'rnn'、'lstm' 或 'gru'")
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        cell_class = {"rnn": RNNCell, "lstm": LSTMCell, "gru": GRUCell}[cell_type]
        self.cell = cell_class(embedding_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, tokens: Tensor, state: Tensor | tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor]]:
        if tokens.ndim != 2 or tokens.size(1) == 0:
            raise ValueError("tokens 应为非空的 [batch, time] 整数张量")
        batch_size = tokens.size(0)
        if state is None:
            zeros = self.embedding.weight.new_zeros(batch_size, self.hidden_size)
            state = (zeros, zeros.clone()) if self.cell_type == "lstm" else zeros

        outputs = []
        for embedding_t in self.embedding(tokens).unbind(dim=1):
            state = self.cell(embedding_t, state)
            hidden = state[0] if self.cell_type == "lstm" else state
            outputs.append(self.output(hidden))
        return torch.stack(outputs, dim=1), state


if __name__ == "__main__":
    for name in ("rnn", "lstm", "gru"):
        model = RecurrentLanguageModel(vocab_size=20, cell_type=name)
        logits, _ = model(torch.randint(0, 20, (2, 5)))
        print(name, tuple(logits.shape))
