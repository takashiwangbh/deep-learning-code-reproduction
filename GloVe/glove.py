"""GloVe weighted least-squares objective over co-occurrence counts."""

import torch
from torch import nn


class GloVe(nn.Module):
    def __init__(self, vocabulary_size: int, width: int, x_max: float = 100.0, alpha: float = 0.75):
        super().__init__()
        self.word = nn.Embedding(vocabulary_size, width)
        self.context = nn.Embedding(vocabulary_size, width)
        self.word_bias = nn.Embedding(vocabulary_size, 1)
        self.context_bias = nn.Embedding(vocabulary_size, 1)
        for embedding in (self.word, self.context):
            nn.init.normal_(embedding.weight, std=0.01)
        nn.init.zeros_(self.word_bias.weight)
        nn.init.zeros_(self.context_bias.weight)
        self.x_max, self.alpha = x_max, alpha

    def forward(self, word_ids: torch.Tensor, context_ids: torch.Tensor, count: torch.Tensor):
        if torch.any(count <= 0):
            raise ValueError("co-occurrence counts must be positive")
        prediction = (self.word(word_ids) * self.context(context_ids)).sum(-1)
        prediction = prediction + self.word_bias(word_ids).squeeze(-1) + self.context_bias(context_ids).squeeze(-1)
        weighting = torch.clamp(count / self.x_max, max=1).pow(self.alpha)
        return (weighting * (prediction - count.log()).square()).mean()

    def word_vectors(self) -> torch.Tensor:
        return self.word.weight + self.context.weight


if __name__ == "__main__":
    model = GloVe(20, 8)
    loss = model(torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5., 2.]))
    loss.backward()
    print("GloVe loss:", float(loss), "vectors:", tuple(model.word_vectors().shape))
