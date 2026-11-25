"""Kim TextCNN: parallel temporal filters followed by max-over-time pooling."""

import torch
from torch import nn
from torch.nn import functional as F


class TextCNN(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int, classes: int,
                 filter_sizes=(2, 3, 4), filters_per_size: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_size, filters_per_size, kernel_size=size) for size in filter_sizes
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(filters_per_size * len(filter_sizes), classes)
        self.minimum_length = max(filter_sizes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] < self.minimum_length:
            raise ValueError("tokens must be [B,T] and T must fit the widest filter")
        x = self.embedding(tokens).transpose(1, 2)
        pooled = [F.relu(conv(x)).amax(dim=-1) for conv in self.convolutions]
        return self.classifier(self.dropout(torch.cat(pooled, dim=-1)))


if __name__ == "__main__":
    model = TextCNN(30, 8, 3)
    y = model(torch.randint(0, 30, (2, 7)))
    y.sum().backward()
    print("TextCNN logits:", tuple(y.shape))
