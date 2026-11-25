"""Small BLIP-style vision-language encoder/decoder with ITC, ITM and LM losses."""

import torch
from torch import nn
from torch.nn import functional as F


class VisionEncoder(nn.Module):
    def __init__(self, image_size: int = 32, patch: int = 8, dim: int = 64):
        super().__init__()
        self.patches = nn.Conv2d(3, dim, patch, stride=patch)
        self.position = nn.Parameter(torch.randn(1, (image_size // patch) ** 2, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, dropout=0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1)

    def forward(self, images: torch.Tensor):
        tokens = self.patches(images).flatten(2).transpose(1, 2)
        if tokens.shape[1] != self.position.shape[1]:
            raise ValueError("Expected the configured image size")
        return self.encoder(tokens + self.position)


class BLIP(nn.Module):
    def __init__(self, vocab_size: int = 100, dim: int = 64, max_length: int = 16,
                 image_size: int = 32, patch: int = 8):
        super().__init__()
        self.visual = VisionEncoder(image_size, patch, dim)
        self.words = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.position = nn.Parameter(torch.randn(1, max_length, dim) * 0.02)
        text_layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, dropout=0, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(dim, 4, dim * 2, dropout=0, batch_first=True)
        self.text_encoder = nn.TransformerEncoder(text_layer, 1)
        self.text_decoder = nn.TransformerDecoder(decoder_layer, 1)
        self.cross_attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.image_projector = nn.Linear(dim, dim)
        self.text_projector = nn.Linear(dim, dim)
        self.matching_head = nn.Linear(dim, 2)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def encode_text(self, tokens: torch.Tensor):
        if tokens.shape[1] > self.position.shape[1]:
            raise ValueError("Text longer than configured max_length")
        embedded = self.words(tokens) + self.position[:, :tokens.shape[1]]
        return self.text_encoder(embedded, src_key_padding_mask=tokens.eq(0))

    def match(self, image_features: torch.Tensor, text_features: torch.Tensor):
        fused, _ = self.cross_attention(text_features, image_features, image_features,
                                        need_weights=False)
        return self.matching_head((text_features + fused)[:, 0])

    def caption_logits(self, image_features: torch.Tensor, input_ids: torch.Tensor):
        length = input_ids.shape[1]
        embedded = self.words(input_ids) + self.position[:, :length]
        causal = torch.ones(length, length, device=input_ids.device, dtype=torch.bool).triu(1)
        decoded = self.text_decoder(embedded, image_features, tgt_mask=causal,
                                    tgt_key_padding_mask=input_ids.eq(0))
        return self.lm_head(decoded)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor):
        """images [B,3,H,W], BOS...EOS/PAD tokens [B,T], B>=2 for negatives."""
        if images.shape[0] < 2 or tokens.shape[1] < 2:
            raise ValueError("Need at least two image-text pairs and two tokens")
        visual = self.visual(images)
        text = self.encode_text(tokens)
        image_vector = F.normalize(self.image_projector(visual.mean(dim=1)), dim=-1)
        text_vector = F.normalize(self.text_projector(text[:, 0]), dim=-1)
        similarity = image_vector @ text_vector.T / 0.07
        labels = torch.arange(images.shape[0], device=images.device)
        itc = 0.5 * (F.cross_entropy(similarity, labels)
                     + F.cross_entropy(similarity.T, labels))

        positive = self.match(visual, text)
        negative = self.match(visual, self.encode_text(tokens.roll(1, dims=0)))
        matching_logits = torch.cat((positive, negative))
        matching_labels = torch.cat((torch.ones(images.shape[0], device=images.device),
                                     torch.zeros(images.shape[0], device=images.device))).long()
        itm = F.cross_entropy(matching_logits, matching_labels)

        lm_logits = self.caption_logits(visual, tokens[:, :-1])
        lm = F.cross_entropy(lm_logits.reshape(-1, lm_logits.shape[-1]),
                             tokens[:, 1:].reshape(-1), ignore_index=0)
        return {"itc": itc, "itm": itm, "lm": lm, "matching_logits": matching_logits}

    @torch.no_grad()
    def generate(self, images: torch.Tensor, bos_id: int = 1, length: int = 8):
        if not 1 <= length <= self.position.shape[1]:
            raise ValueError("Generation length exceeds configured max_length")
        was_training = self.training
        self.eval()
        memory = self.visual(images)
        tokens = torch.full((images.shape[0], 1), bos_id, device=images.device, dtype=torch.long)
        for _ in range(length - 1):
            next_token = self.caption_logits(memory, tokens)[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat((tokens, next_token), dim=-1)
        self.train(was_training)
        return tokens

    @torch.no_grad()
    def bootstrap_and_filter(self, images: torch.Tensor, bos_id: int = 1,
                             length: int = 8, threshold: float = 0.5):
        """CapFilt outline: synthesize captions, retain high ITM probability pairs."""
        captions = self.generate(images, bos_id, length)
        was_training = self.training
        self.eval()
        scores = self.match(self.visual(images), self.encode_text(captions)).softmax(-1)[:, 1]
        self.train(was_training)
        return captions, scores >= threshold, scores


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BLIP(vocab_size=32, dim=32)
    images = torch.randn(3, 3, 32, 32)
    tokens = torch.tensor([[1, 3, 4, 2, 0], [1, 5, 6, 7, 2], [1, 8, 9, 2, 0]])
    losses = model(images, tokens)
    (losses["itc"] + losses["itm"] + losses["lm"]).backward()
    print("BLIP ITC/ITM/LM:", *(round(losses[key].item(), 4) for key in ("itc", "itm", "lm")))
