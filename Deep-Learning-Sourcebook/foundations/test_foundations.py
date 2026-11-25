"""运行：python test_foundations.py。用数值微分验证手写反向传播。"""

import unittest

import numpy as np
import torch

from mlp_numpy import TwoLayerMLP, train_xor
from normalization import BatchNorm1dFromScratch, LayerNormFromScratch
from optimizers_numpy import Adam, SGD


class FoundationTests(unittest.TestCase):
    def test_mlp_gradients_match_finite_difference(self):
        rng = np.random.default_rng(2)
        model = TwoLayerMLP(3, 4, 2)
        x = rng.normal(size=(5, 3))
        y = np.array([0, 1, 1, 0, 1])
        _, gradients = model.loss_and_gradients(x, y)
        eps = 1e-5
        for name, parameter in model.parameters.items():
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + eps
                plus = model.loss_and_gradients(x, y)[0]
                parameter[index] = original - eps
                minus = model.loss_and_gradients(x, y)[0]
                parameter[index] = original
                numeric = (plus - minus) / (2 * eps)
                self.assertAlmostEqual(numeric, gradients[name][index], places=6)

    def test_xor_learns_non_linear_boundary(self):
        model, loss = train_xor()
        x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        np.testing.assert_array_equal(model.predict(x), [0, 1, 1, 0])
        self.assertLess(loss, 0.01)

    def test_optimizer_updates(self):
        parameter = {"x": np.array([1.0])}
        SGD(0.1).step(parameter, {"x": np.array([2.0])})
        np.testing.assert_allclose(parameter["x"], [0.8])
        parameter = {"x": np.array([1.0])}
        Adam(0.1).step(parameter, {"x": np.array([2.0])})
        np.testing.assert_allclose(parameter["x"], [0.9], rtol=1e-7)

    def test_normalization_statistics_and_gradients(self):
        torch.manual_seed(2)
        x = torch.randn(8, 4, requires_grad=True)
        bn = BatchNorm1dFromScratch(4)
        y = bn(x)
        torch.testing.assert_close(y.mean(dim=0), torch.zeros(4), atol=1e-6, rtol=0)
        y.square().sum().backward()
        self.assertIsNotNone(x.grad)
        bn.eval()
        self.assertEqual(tuple(bn(torch.randn(3, 4)).shape), (3, 4))
        ln = LayerNormFromScratch(4)
        z = ln(torch.randn(2, 3, 4))
        torch.testing.assert_close(z.mean(dim=-1), torch.zeros(2, 3), atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
