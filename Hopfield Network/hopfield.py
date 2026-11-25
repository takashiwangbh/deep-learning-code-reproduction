"""Classical Hopfield associative memory with asynchronous energy descent."""

import torch
from torch import nn


class HopfieldMemory(nn.Module):
    def __init__(self, patterns: torch.Tensor):
        super().__init__()
        if patterns.ndim != 2 or not torch.all((patterns == -1) | (patterns == 1)):
            raise ValueError("patterns must be [memories,neurons] bipolar -1/+1")
        _, neurons = patterns.shape
        weights = patterns.T @ patterns / neurons
        weights.fill_diagonal_(0)
        self.register_buffer("weights", weights)

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """Energy for each [neurons] state; lower is preferred."""
        return -0.5 * (state * (state @ self.weights)).sum(dim=-1)

    @torch.no_grad()
    def recall(self, noisy: torch.Tensor, sweeps: int = 5) -> torch.Tensor:
        if noisy.shape[-1] != self.weights.shape[0]:
            raise ValueError("state width differs from stored memories")
        state = noisy.clone()
        for _ in range(sweeps):
            for neuron in range(self.weights.shape[0]):
                field = state @ self.weights[:, neuron]
                state[..., neuron] = torch.where(field > 0, 1.,
                                                  torch.where(field < 0, -1., state[..., neuron]))
        return state


if __name__ == "__main__":
    patterns = torch.tensor([[1., 1., -1., -1., 1., 1.]])
    memory = HopfieldMemory(patterns)
    noisy = patterns.clone()
    noisy[0, 2] *= -1
    recovered = memory.recall(noisy)
    print("Energy before/after:", memory.energy(noisy).item(),
          memory.energy(recovered).item(), "state:", recovered.int().tolist())
