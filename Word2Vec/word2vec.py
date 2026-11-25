"""Skip-gram with negative sampling (SGNS), including its training loss."""

import torch
from torch import nn
from torch.nn import functional as F


class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocabulary_size: int, width: int):
        super().__init__()
        self.target_embeddings = nn.Embedding(vocabulary_size, width)
        self.context_embeddings = nn.Embedding(vocabulary_size, width)
        nn.init.uniform_(self.target_embeddings.weight, -0.5 / width, 0.5 / width)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(self, center: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor):
        """center/positive [B], negatives [B,K]."""
        u = self.target_embeddings(center)
        v = self.context_embeddings(positive)
        noise = self.context_embeddings(negatives)
        positive_score = (u * v).sum(-1)
        negative_score = torch.bmm(noise, u.unsqueeze(-1)).squeeze(-1)
        return -(F.logsigmoid(positive_score) + F.logsigmoid(-negative_score).sum(-1)).mean()

    def word_vectors(self) -> torch.Tensor:
        return self.target_embeddings.weight


if __name__ == "__main__":
    model = SkipGramNegativeSampling(20, 8)
    loss = model(torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([[5, 6], [7, 8]]))
    loss.backward()
    print("SGNS loss:", float(loss), "vectors:", tuple(model.word_vectors().shape))
