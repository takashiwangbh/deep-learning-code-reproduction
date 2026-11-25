"""Small TFT: static context, variable selection, local LSTM, gated skip and causal attention."""

import torch
from torch import nn
from torch.nn import functional as F


class VariableSelection(nn.Module):
    def __init__(self, variables: int, width: int):
        super().__init__()
        self.projections = nn.ModuleList(nn.Linear(1, width) for _ in range(variables))
        self.selection = nn.Sequential(nn.Linear(variables + width, width), nn.ELU(),
                                       nn.Linear(width, variables))

    def forward(self, values: torch.Tensor, context: torch.Tensor):
        """values [B,T,V], static context [B,H] -> selected [B,T,H]."""
        batch, time, _ = values.shape
        static = context[:, None, :].expand(batch, time, -1)
        weights = self.selection(torch.cat((values, static), dim=-1)).softmax(-1)
        embedded = torch.stack([layer(values[..., index:index + 1])
                                for index, layer in enumerate(self.projections)], dim=-2)
        return (weights[..., None] * embedded).sum(-2), weights


class GatedResidual(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.transform = nn.Sequential(nn.Linear(width, width), nn.ELU(), nn.Linear(width, width))
        self.gate = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, residual: torch.Tensor):
        return self.norm(residual + torch.sigmoid(self.gate(x)) * self.transform(x))


class TemporalFusionTransformer(nn.Module):
    def __init__(self, static_features: int, past_variables: int, future_variables: int,
                 width: int = 32, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = tuple(quantiles)
        self.static_context = nn.Sequential(nn.Linear(static_features, width), nn.ELU(),
                                            nn.Linear(width, width))
        self.past_selection = VariableSelection(past_variables, width)
        self.future_selection = VariableSelection(future_variables, width)
        self.initial_hidden = nn.Linear(width, width)
        self.initial_cell = nn.Linear(width, width)
        self.past_lstm = nn.LSTM(width, width, batch_first=True)
        self.future_lstm = nn.LSTM(width, width, batch_first=True)
        self.local_gate = GatedResidual(width)
        self.attention = nn.MultiheadAttention(width, 1, batch_first=True, dropout=0)
        self.attention_gate = GatedResidual(width)
        self.output = nn.Linear(width, len(self.quantiles))

    def forward(self, static: torch.Tensor, past: torch.Tensor, known_future: torch.Tensor):
        """static[B,S], past[B,P,Vp], future[B,F,Vf] -> quantiles[B,F,Q]."""
        context = self.static_context(static)
        selected_past, past_weights = self.past_selection(past, context)
        selected_future, future_weights = self.future_selection(known_future, context)
        initial = (self.initial_hidden(context).tanh()[None],
                   self.initial_cell(context).tanh()[None])
        historical, state = self.past_lstm(selected_past, initial)
        prospective, _ = self.future_lstm(selected_future, state)
        local = self.local_gate(torch.cat((historical, prospective), dim=1),
                                torch.cat((selected_past, selected_future), dim=1))
        past_length, future_length = past.shape[1], known_future.shape[1]
        positions = torch.arange(past_length + future_length, device=local.device)
        future_positions = positions[past_length:]
        # Forecast step j may see all history and known future up to j, never later steps.
        causal = positions[None, :] > future_positions[:, None]
        attended, weights = self.attention(local[:, past_length:], local, local,
                                            attn_mask=causal, need_weights=True)
        fused = self.attention_gate(attended, local[:, past_length:])
        return self.output(fused), {"past_selection": past_weights,
                                    "future_selection": future_weights,
                                    "temporal_attention": weights}


def quantile_loss(prediction: torch.Tensor, target: torch.Tensor, quantiles=(0.1, 0.5, 0.9)):
    errors = target[..., None] - prediction
    q = torch.tensor(quantiles, device=prediction.device, dtype=prediction.dtype)
    return torch.maximum(q * errors, (q - 1) * errors).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TemporalFusionTransformer(2, 3, 2, width=16)
    output, interpretation = model(torch.randn(2, 2), torch.randn(2, 6, 3),
                                    torch.randn(2, 4, 2))
    quantile_loss(output, torch.randn(2, 4)).backward()
    print("TFT quantiles/attention:", tuple(output.shape),
          tuple(interpretation["temporal_attention"].shape))
