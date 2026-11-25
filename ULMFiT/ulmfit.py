"""ULMFiT teaching core: language model, concat-pooling head, gradual unfreezing and LR schedule."""

import torch
from torch import nn
from torch.nn import functional as F


class ULMFiT(nn.Module):
    def __init__(self, vocab: int, classes: int, width: int = 32, layers: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.recurrent = nn.ModuleList(nn.LSTM(width, width, batch_first=True)
                                       for _ in range(layers))
        self.lm_head = nn.Linear(width, vocab)
        self.classifier = nn.Sequential(nn.Linear(3 * width, width), nn.ReLU(),
                                        nn.Linear(width, classes))

    def encode(self, ids: torch.Tensor):
        hidden = self.embedding(ids)
        for layer in self.recurrent:
            hidden, _ = layer(hidden)
        return hidden

    def language_model_loss(self, ids: torch.Tensor):
        if ids.shape[1] < 2:
            raise ValueError("Language modeling needs at least two tokens")
        logits = self.lm_head(self.encode(ids[:, :-1]))
        return F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten())

    def classify(self, ids: torch.Tensor):
        hidden = self.encode(ids)
        last = hidden[:, -1]
        pooled = torch.cat((last, hidden.max(dim=1).values, hidden.mean(dim=1)), dim=-1)
        return self.classifier(pooled)

    def gradual_unfreeze(self, stage: int):
        """0=head only, 1=last LSTM, ..., layers+1=all LM layers and embeddings."""
        if not 0 <= stage <= len(self.recurrent) + 1:
            raise ValueError("Stage outside available encoder layers")
        for param in self.embedding.parameters():
            param.requires_grad_(stage == len(self.recurrent) + 1)
        for index, layer in enumerate(self.recurrent):
            trainable = stage >= len(self.recurrent) - index
            for param in layer.parameters():
                param.requires_grad_(trainable)
        for param in self.lm_head.parameters():
            param.requires_grad_(False)  # LM head is not needed for classification
        for param in self.classifier.parameters():
            param.requires_grad_(True)

    def discriminative_groups(self, top_lr: float, decay: float = 2.6):
        """Fresh optimizer parameter groups; deeper encoder layers get larger LR."""
        layers = [self.embedding, *self.recurrent, self.classifier]
        return [{"params": list(layer.parameters()), "lr": top_lr / decay ** (len(layers) - 1 - index)}
                for index, layer in enumerate(layers)
                if any(param.requires_grad for param in layer.parameters())]


def slanted_triangular_lr(step: int, total_steps: int, cut_fraction: float = .1,
                          ratio: float = 32.) -> float:
    """Multiplier in [1/ratio,1], rising rapidly then falling gradually."""
    if total_steps < 2 or not 0 <= step <= total_steps or not 0 < cut_fraction < 1:
        raise ValueError("Invalid schedule range")
    cut = max(1, int(total_steps * cut_fraction))
    progress = step / cut if step < cut else 1 - (step - cut) / (total_steps - cut)
    return (1 + progress * (ratio - 1)) / ratio


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ULMFiT(vocab=30, classes=3, width=12)
    ids = torch.randint(0, 30, (2, 6))
    lm_loss = model.language_model_loss(ids)
    model.gradual_unfreeze(stage=1)
    logits = model.classify(ids)
    logits.sum().backward()
    print("ULMFiT classification/LM loss:", tuple(logits.shape), round(lm_loss.item(), 4))
