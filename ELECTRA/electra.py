"""ELECTRA: masked generator samples tokens; discriminator detects actual replacements."""

import torch
from torch import nn
from torch.nn import functional as F


class TextEncoder(nn.Module):
    def __init__(self, vocab: int, length: int, width: int, heads: int = 4):
        super().__init__()
        self.words = nn.Embedding(vocab, width, padding_idx=0)
        self.positions = nn.Embedding(length, width)
        block = nn.TransformerEncoderLayer(width, heads, width * 4,
                                           dropout=0, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, 2)

    def forward(self, ids: torch.Tensor):
        batch, time = ids.shape
        if time > self.positions.num_embeddings:
            raise ValueError("Sequence exceeds configured length")
        embedded = self.words(ids) + self.positions(torch.arange(time, device=ids.device))[None]
        return self.encoder(embedded, src_key_padding_mask=ids.eq(0))


class ELECTRA(nn.Module):
    def __init__(self, vocab: int = 64, length: int = 32,
                 generator_width: int = 16, discriminator_width: int = 32):
        super().__init__()
        if vocab <= 4:
            raise ValueError("Vocabulary reserves IDs 0..3")
        self.vocab = vocab
        self.generator = TextEncoder(vocab, length, generator_width)
        self.discriminator = TextEncoder(vocab, length, discriminator_width)
        self.generator_head = nn.Linear(generator_width, vocab)
        self.discriminator_head = nn.Linear(discriminator_width, 1)

    def training_loss(self, clean: torch.Tensor, mask_probability: float = .15,
                      discriminator_weight: float = 50.0):
        """0=PAD, 1=MASK, 2/3=special; clean [B,T]."""
        eligible = clean.ge(4)
        if not eligible.any(dim=1).all():
            raise ValueError("Every example needs a non-special token")
        selected = (torch.rand(clean.shape, device=clean.device) < mask_probability) & eligible
        for batch in range(clean.shape[0]):
            if not selected[batch].any():
                selected[batch, eligible[batch].nonzero()[0].item()] = True
        masked = clean.masked_fill(selected, 1)
        generator_logits = self.generator_head(self.generator(masked))
        generator_loss = F.cross_entropy(generator_logits[selected], clean[selected])

        with torch.no_grad():
            possible = generator_logits[selected].detach().clone()
            possible[:, :4] = torch.finfo(possible.dtype).min
            sampled = possible.softmax(dim=-1).multinomial(1).squeeze(-1)
        corrupted = clean.clone()
        corrupted[selected] = sampled
        # Sampled original IDs count as original, not as fake replacements.
        replaced = corrupted.ne(clean)
        discriminator_logits = self.discriminator_head(self.discriminator(corrupted)).squeeze(-1)
        valid = clean.ne(0)
        discriminator_loss = F.binary_cross_entropy_with_logits(
            discriminator_logits[valid], replaced[valid].float())
        return generator_loss + discriminator_weight * discriminator_loss, {
            "generator": generator_loss, "discriminator": discriminator_loss,
            "corrupted": corrupted, "replaced": replaced, "masked": selected,
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ELECTRA(vocab=32, length=8)
    ids = torch.tensor([[2, 4, 5, 6, 7, 3], [2, 8, 9, 10, 11, 3]])
    loss, details = model.training_loss(ids)
    loss.backward()
    print("ELECTRA masked/replaced/loss:", int(details["masked"].sum()),
          int(details["replaced"].sum()), round(loss.item(), 4))
