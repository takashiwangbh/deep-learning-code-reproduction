"""Deep Interest Network: candidate-dependent activation of behavior history."""

import torch
from torch import nn


class LocalActivation(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * width, width), nn.PReLU(),
                                 nn.Linear(width, width // 2), nn.PReLU(),
                                 nn.Linear(width // 2, 1))

    def forward(self, target: torch.Tensor, history: torch.Tensor):
        target = target[:, None, :].expand_as(history)
        combined = torch.cat((target, history, target - history, target * history), dim=-1)
        return torch.sigmoid(self.net(combined).squeeze(-1))


class DIN(nn.Module):
    def __init__(self, items: int, width: int = 32):
        super().__init__()
        self.items = nn.Embedding(items, width, padding_idx=0)
        self.activate = LocalActivation(width)
        self.head = nn.Sequential(nn.Linear(2 * width, width), nn.PReLU(),
                                  nn.Linear(width, 1))

    def forward(self, history_ids: torch.Tensor, target_ids: torch.Tensor):
        """History [B,T] (zero is PAD), target [B] -> click logits [B]."""
        history = self.items(history_ids)
        target = self.items(target_ids)
        weights = self.activate(target, history) * history_ids.ne(0)
        interest = (history * weights[..., None]).sum(dim=1)
        logits = self.head(torch.cat((interest, target), dim=-1)).squeeze(-1)
        return logits, weights


if __name__ == "__main__":
    torch.manual_seed(0)
    model = DIN(items=30, width=16)
    history = torch.tensor([[1, 2, 3, 0], [5, 6, 0, 0]])
    logit, attention = model(history, torch.tensor([7, 8]))
    logit.sum().backward()
    print("DIN logits/interest weights:", tuple(logit.shape), tuple(attention.shape))
