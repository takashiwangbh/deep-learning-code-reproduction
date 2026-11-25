"""Verify the efficient FM identity against explicit pair enumeration."""

import importlib.util
from pathlib import Path
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[1] / "Factorization Machines" / "factorization_machines.py"
SPEC = importlib.util.spec_from_file_location("factorization_machines", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FactorizationMachineTests(unittest.TestCase):
    def test_pairwise_identity_and_gradients(self):
        torch.manual_seed(0)
        model = MODULE.FactorizationMachine(features=5, factors=3)
        x = torch.randn(4, 5)
        efficient = model(x)
        explicit = model.bias + x @ model.linear
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                explicit = explicit + x[:, i] * x[:, j] * torch.dot(
                    model.factors[i], model.factors[j])
        self.assertTrue(torch.allclose(efficient, explicit, atol=1e-6))
        efficient.sum().backward()
        self.assertTrue(torch.isfinite(model.factors.grad).all())


if __name__ == "__main__":
    unittest.main()
