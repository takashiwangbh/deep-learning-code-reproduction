"""RoBERTa teaching core: bidirectional encoder with fresh 80/10/10 MLM masks per batch."""

import torch
from torch import nn
from torch.nn import functional as F


def dynamic_mask(ids: torch.Tensor, vocab: int, mask_id: int = 1,
                 protected=(0, 1, 2, 3), probability: float = .15):
    """Return corrupted tokens and original labels only at selected positions."""
    eligible = torch.ones_like(ids, dtype=torch.bool)
    for special in protected:
        eligible &= ids.ne(special)
    if not eligible.any(dim=1).all():
        raise ValueError("Each sample needs at least one non-special token")
    selected = (torch.rand(ids.shape, device=ids.device) < probability) & eligible
    for batch in range(ids.shape[0]):
        if not selected[batch].any():
            selected[batch, eligible[batch].nonzero()[0]] = True
    labels = ids.masked_fill(~selected, -100)
    choice = torch.rand(ids.shape, device=ids.device)
    corrupted = ids.clone()
    corrupted[selected & (choice < .8)] = mask_id
    random_positions = selected & (choice >= .8) & (choice < .9)
    random_tokens = torch.randint(4, vocab, ids.shape, device=ids.device)
    corrupted[random_positions] = random_tokens[random_positions]
    return corrupted, labels


class RoBERTa(nn.Module):
    def __init__(self, vocab: int = 64, length: int = 32, width: int = 32,
                 heads: int = 4, layers: int = 2):
        super().__init__()
        if vocab <= 4:
            raise ValueError("Vocabulary must reserve four special IDs")
        self.vocab = vocab
        self.words = nn.Embedding(vocab, width, padding_idx=0)
        self.positions = nn.Embedding(length, width)
        block = nn.TransformerEncoderLayer(width, heads, width * 4,
                                           dropout=0, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.lm_head = nn.Linear(width, vocab)

    def forward(self, ids: torch.Tensor):
        batch, time = ids.shape
        if time > self.positions.num_embeddings:
            raise ValueError("Sequence longer than configured maximum")
        position = torch.arange(time, device=ids.device)
        features = self.words(ids) + self.positions(position)[None]
        encoded = self.encoder(features, src_key_padding_mask=ids.eq(0))
        return self.lm_head(encoded)

    def training_loss(self, clean_ids: torch.Tensor):
        corrupted, labels = dynamic_mask(clean_ids, self.vocab)
        logits = self(corrupted)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab), labels.reshape(-1),
                               ignore_index=-100)
        return loss, corrupted, labels


if __name__ == "__main__":
    torch.manual_seed(0)
    model = RoBERTa(vocab=32, length=8, width=16)
    clean = torch.tensor([[2, 4, 5, 6, 7, 3], [2, 8, 9, 10, 11, 3]])
    loss, masked, targets = model.training_loss(clean)
    loss.backward()
    print("RoBERTa MLM loss/selected tokens:", round(loss.item(), 4), int(targets.ne(-100).sum()))
