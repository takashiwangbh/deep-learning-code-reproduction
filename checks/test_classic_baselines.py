"""Mechanism checks for historical and vision baseline projects."""

import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def load(folder: str, filename: str):
    path = ROOT / folder / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClassicBaselineTests(unittest.TestCase):
    def test_hopfield_energy_descends(self):
        module = load("Hopfield Network", "hopfield.py")
        patterns = torch.tensor([[1., 1., -1., -1., 1., 1.]])
        memory = module.HopfieldMemory(patterns)
        noisy = patterns.clone()
        noisy[0, 2] = 1
        recovered = memory.recall(noisy)
        self.assertTrue(torch.allclose(memory.weights, memory.weights.T))
        self.assertTrue(torch.equal(memory.weights.diag(), torch.zeros(6)))
        self.assertLessEqual(memory.energy(recovered).item(), memory.energy(noisy).item())
        self.assertTrue(torch.equal(recovered, patterns))

    def test_rbm_free_energy_and_dbn_generation(self):
        rbm_module = load("Restricted Boltzmann Machine", "rbm.py")
        rbm = rbm_module.BernoulliRBM(2, 2)
        data = torch.tensor([[1., 0.], [0., 1.]])
        hidden_states = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        v = data[0]
        energies = -(v * rbm.visible_bias).sum() - (hidden_states * rbm.hidden_bias).sum(-1)
        energies -= (v @ rbm.weight * hidden_states).sum(-1)
        self.assertTrue(torch.allclose(rbm.free_energy(v), -torch.logsumexp(-energies, 0)))
        loss = rbm.cd_loss(data, steps=2)
        loss.backward()
        self.assertIsNotNone(rbm.weight.grad)
        dbn_module = load("Deep Belief Network", "dbn.py")
        dbn = dbn_module.DeepBeliefNetwork([6, 4, 3])
        binary = torch.randint(0, 2, (12, 6)).float()
        history = dbn.greedy_pretrain(binary, epochs=2)
        self.assertEqual(len(history), 2)
        self.assertTrue(torch.isfinite(torch.tensor(history)).all())
        self.assertEqual(tuple(dbn.encode(binary).shape), (12, 3))
        sample = dbn.generate(4)
        self.assertEqual(tuple(sample.shape), (4, 6))
        self.assertTrue(torch.all((sample == 0) | (sample == 1)))

    def test_dropout_expectation_and_highway_carry(self):
        dropout = load("Dropout", "dropout.py")
        highway = load("Highway Network", "highway.py")
        torch.manual_seed(0)
        ones = torch.ones(100000)
        dropped = dropout.inverted_dropout(ones, .3, True)
        self.assertAlmostEqual(dropped.mean().item(), 1., delta=.02)
        self.assertTrue(torch.equal(dropout.inverted_dropout(ones, .3, False), ones))
        layer = highway.HighwayLayer(5)
        with torch.no_grad():
            layer.gate.weight.zero_()
            layer.gate.bias.fill_(-30)
        x = torch.randn(3, 5)
        self.assertTrue(torch.allclose(layer(x), x, atol=1e-5))

    def test_resnet50_bottleneck_and_fpn_topdown(self):
        resnet = load("ResNet-50", "resnet50.py")
        fpn = load("Feature Pyramid Network", "fpn.py")
        model = resnet.ResNet50(classes=3, base_width=8)
        self.assertEqual([len(stage) for stage in model.stages], [3, 4, 6, 3])
        logits = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (2, 3))
        logits.mean().backward()
        self.assertIsNotNone(model.stages[-1][-1].main[0].weight.grad)
        pyramid = fpn.FeaturePyramidNetwork((8, 16, 24, 32), width=12)
        maps = [torch.randn(1, c, h, h, requires_grad=True)
                for c, h in zip((8, 16, 24, 32), (16, 8, 4, 2))]
        outputs = pyramid(maps)
        self.assertEqual([outputs[f"P{i}"].shape[-1] for i in range(2, 7)], [16, 8, 4, 2, 1])
        outputs["P2"].sum().backward()
        self.assertGreater(maps[-1].grad.abs().sum().item(), 0)

    def test_pspnet_and_pointnetplusplus(self):
        psp = load("PSPNet", "pspnet.py")
        pointnet = load("PointNet++", "pointnet_plus_plus.py")
        image = torch.randn(2, 3, 48, 40)
        segmenter = psp.PSPNet(classes=3, width=16)
        logits = segmenter(image)
        self.assertEqual(tuple(logits.shape), (2, 3, 48, 40))
        logits.mean().backward()
        self.assertTrue(all(branch.weight.grad is not None for branch in segmenter.context.branches))
        torch.manual_seed(4)
        xyz = torch.rand(2, 32, 3)
        permutation = torch.randperm(32)
        model = pointnet.PointNetPlusPlus(classes=5)
        model.eval()
        original = model(xyz)
        shuffled = model(xyz[:, permutation])
        self.assertEqual(tuple(original.shape), (2, 5))
        self.assertTrue(torch.allclose(original, shuffled, atol=1e-5))
        original.mean().backward()
        self.assertIsNotNone(model.first.local_mlp[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
