"""Bahdanau 加性注意力的编码器-解码器，源码可直接阅读。

运行：python 'Attention Seq2Seq/attention_seq2seq.py'。GRU 单元源码在同目录。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from gru_cell import GRUCell


class AdditiveAttention(nn.Module):
    """e_(t,s) = v^T tanh(W_q h_t + W_k h_s)，对源序列做 softmax。"""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query: Tensor, values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        # query: [B,H]，values: [B,S,H]，valid: [B,S]。
        energies = self.score(
            torch.tanh(self.query(query).unsqueeze(1) + self.key(values))
        ).squeeze(-1)
        energies = energies.masked_fill(~valid, torch.finfo(energies.dtype).min)
        weights = torch.softmax(energies, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
        return context, weights


class AttentionSeq2Seq(nn.Module):
    """GRU 编码器 + 加性注意力 + GRU 解码器。

    src: [B,S]，tgt_input: [B,T]，返回下一词 logits: [B,T,V_tgt]。
    tgt_input 应以 BOS 开头；训练标签是同一目标句右移后的 token。
    """

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        embedding_size: int = 64,
        hidden_size: int = 128,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.pad_id = pad_id
        self.source_embedding = nn.Embedding(source_vocab_size, embedding_size)
        self.target_embedding = nn.Embedding(target_vocab_size, embedding_size)
        self.encoder = GRUCell(embedding_size, hidden_size)
        self.attention = AdditiveAttention(hidden_size)
        self.decoder = GRUCell(embedding_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size + hidden_size, target_vocab_size)

    def encode(self, source: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if source.ndim != 2 or source.size(1) == 0:
            raise ValueError("source 应为非空的 [batch, source_time] 张量")
        valid = source.ne(self.pad_id)
        if not bool(valid.any(dim=1).all()):
            raise ValueError("每个源序列至少需要一个非 PAD token")
        state = self.source_embedding.weight.new_zeros(source.size(0), self.hidden_size)
        states = []
        for step, embedding_t in enumerate(self.source_embedding(source).unbind(dim=1)):
            new_state = self.encoder(embedding_t, state)
            # PAD 之后保持最后一个真实 token 的状态。
            state = torch.where(valid[:, step, None], new_state, state)
            states.append(state)
        return torch.stack(states, dim=1), state, valid

    def decode_step(
        self, token: Tensor, state: Tensor, encoder_states: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        context, weights = self.attention(state, encoder_states, valid)
        decoder_input = torch.cat((self.target_embedding(token), context), dim=-1)
        state = self.decoder(decoder_input, state)
        logits = self.output(torch.cat((state, context), dim=-1))
        return logits, state, weights

    def forward(self, source: Tensor, target_input: Tensor) -> Tensor:
        if target_input.ndim != 2 or target_input.size(1) == 0:
            raise ValueError("target_input 应为非空的 [batch, target_time] 张量")
        encoder_states, state, valid = self.encode(source)
        logits = []
        for token in target_input.unbind(dim=1):
            output_t, state, _ = self.decode_step(token, state, encoder_states, valid)
            logits.append(output_t)
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def greedy_decode(self, source: Tensor, bos_id: int, eos_id: int, max_steps: int) -> Tensor:
        """逐词推理；输出包含 BOS 与生成的 token。"""
        if max_steps <= 0:
            raise ValueError("max_steps 必须大于零")
        encoder_states, state, valid = self.encode(source)
        token = torch.full((source.size(0),), bos_id, dtype=torch.long, device=source.device)
        generated = [token]
        finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        for _ in range(max_steps):
            logits, state, _ = self.decode_step(token, state, encoder_states, valid)
            token = logits.argmax(dim=-1)
            token = torch.where(finished, torch.full_like(token, eos_id), token)
            generated.append(token)
            finished |= token.eq(eos_id)
            if bool(finished.all()):
                break
        return torch.stack(generated, dim=1)


if __name__ == "__main__":
    model = AttentionSeq2Seq(source_vocab_size=30, target_vocab_size=40)
    source = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]])
    target_input = torch.tensor([[1, 7, 8], [1, 9, 10]])
    print("logits:", tuple(model(source, target_input).shape))
