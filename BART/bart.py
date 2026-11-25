"""BART-style span-infilling denoising with bidirectional encoder and causal decoder."""

import torch
from torch import nn
from torch.nn import functional as F


def span_infilling(ids: torch.Tensor, start: int, end: int, mask_id: int = 1):
    """Replace a contiguous span [start,end) with ONE mask token."""
    if not 1 <= start < end <= ids.shape[1] - 1:
        raise ValueError("Keep BOS and EOS outside the corrupted span")
    mask = ids.new_full((ids.shape[0], 1), mask_id)
    return torch.cat((ids[:, :start], mask, ids[:, end:]), dim=1)


class BART(nn.Module):
    def __init__(self, vocab: int = 64, length: int = 32, width: int = 32,
                 heads: int = 4, layers: int = 2):
        super().__init__()
        self.words = nn.Embedding(vocab, width, padding_idx=0)
        self.positions = nn.Embedding(length, width)
        encoder = nn.TransformerEncoderLayer(width, heads, width * 4,
                                             dropout=0, batch_first=True)
        decoder = nn.TransformerDecoderLayer(width, heads, width * 4,
                                             dropout=0, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, layers)
        self.decoder = nn.TransformerDecoder(decoder, layers)
        self.output = nn.Linear(width, vocab)

    def _embed(self, ids: torch.Tensor):
        time = ids.shape[1]
        if time > self.positions.num_embeddings:
            raise ValueError("Text longer than configured maximum")
        return self.words(ids) + self.positions(torch.arange(time, device=ids.device))[None]

    def forward(self, corrupted_source: torch.Tensor, decoder_input: torch.Tensor):
        source_padding = corrupted_source.eq(0)
        target_padding = decoder_input.eq(0)
        memory = self.encoder(self._embed(corrupted_source),
                              src_key_padding_mask=source_padding)
        time = decoder_input.shape[1]
        causal = torch.ones(time, time, device=decoder_input.device, dtype=torch.bool).triu(1)
        hidden = self.decoder(self._embed(decoder_input), memory,
                              tgt_mask=causal, tgt_key_padding_mask=target_padding,
                              memory_key_padding_mask=source_padding)
        return self.output(hidden)

    def denoising_loss(self, clean: torch.Tensor, span_start: int = 2,
                       span_length: int = 2):
        source = span_infilling(clean, span_start, span_start + span_length)
        logits = self(source, clean[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), clean[:, 1:].reshape(-1),
                               ignore_index=0)
        return loss, source


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BART(vocab=32, length=10, width=16)
    clean = torch.tensor([[2, 4, 5, 6, 7, 3], [2, 8, 9, 10, 11, 3]])
    loss, source = model.denoising_loss(clean)
    loss.backward()
    print("BART corrupted/clean lengths:", source.shape[1], clean.shape[1],
          "loss:", round(loss.item(), 4))
