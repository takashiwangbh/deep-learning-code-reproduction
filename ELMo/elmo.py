"""ELMo: separate forward/backward language models and a learned layer mixture."""

import torch
from torch import nn
from torch.nn import functional as F


class ELMo(nn.Module):
    def __init__(self, vocab: int, width: int = 32, layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.forward_layers = nn.ModuleList(nn.LSTM(width, width, batch_first=True)
                                            for _ in range(layers))
        self.backward_layers = nn.ModuleList(nn.LSTM(width, width, batch_first=True)
                                             for _ in range(layers))
        self.forward_head = nn.Linear(width, vocab)
        self.backward_head = nn.Linear(width, vocab)
        self.layer_logits = nn.Parameter(torch.zeros(layers + 1))
        self.gamma = nn.Parameter(torch.ones(()))

    @staticmethod
    def _stack(layers, tokens):
        outputs = []
        for layer in layers:
            tokens, _ = layer(tokens)
            outputs.append(tokens)
        return outputs

    def contextual_embeddings(self, ids: torch.Tensor):
        """ids [B,T] -> task-specific contextual word vectors [B,T,2*width]."""
        embedded = self.embedding(ids)
        forward = self._stack(self.forward_layers, embedded)
        backward = [h.flip(1) for h in self._stack(self.backward_layers, embedded.flip(1))]
        representations = [torch.cat((embedded, embedded), dim=-1)]
        representations += [torch.cat((left, right), dim=-1)
                            for left, right in zip(forward, backward)]
        weights = self.layer_logits.softmax(dim=0)
        return self.gamma * sum(weight * layer
                                for weight, layer in zip(weights, representations)), weights

    def language_model_loss(self, ids: torch.Tensor):
        """Forward predicts next word, backward predicts previous word; no target leakage."""
        if ids.shape[1] < 2:
            raise ValueError("Need at least two tokens")
        forward_input = self.embedding(ids[:, :-1])
        backward_input = self.embedding(ids[:, 1:].flip(1))
        forward_hidden = self._stack(self.forward_layers, forward_input)[-1]
        backward_hidden = self._stack(self.backward_layers, backward_input)[-1]
        forward_labels = ids[:, 1:]
        backward_labels = ids[:, :-1].flip(1)
        loss_forward = F.cross_entropy(self.forward_head(forward_hidden).flatten(0, 1),
                                       forward_labels.flatten())
        loss_backward = F.cross_entropy(self.backward_head(backward_hidden).flatten(0, 1),
                                        backward_labels.flatten())
        return loss_forward + loss_backward


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ELMo(vocab=30, width=12)
    ids = torch.randint(0, 30, (2, 6))
    vectors, mix = model.contextual_embeddings(ids)
    model.language_model_loss(ids).backward()
    print("ELMo contextual/mix:", tuple(vectors.shape), tuple(mix.shape))
