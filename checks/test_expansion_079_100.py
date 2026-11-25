"""Fast CPU checks for graph, 3D, self-supervised, RL and recommendation additions."""

import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def source(folder: str, filename: str):
    path = ROOT / folder / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExpansionTests(unittest.TestCase):
    def test_graph_models(self):
        sage = source("GraphSAGE", "graphsage.py")
        gat = source("GAT", "gat.py")
        gin = source("GIN", "gin.py")
        x = torch.randn(4, 5)
        adjacency = torch.tensor([[0, 1, 0, 0], [1, 0, 1, 0],
                                  [0, 1, 0, 1], [0, 0, 1, 0]])
        self.assertEqual(tuple(sage.GraphSAGE(5, 8, 3)(x, adjacency).shape), (4, 3))
        self.assertEqual(tuple(gat.GAT(5, 4, 2)(x, adjacency).shape), (4, 2))
        self.assertEqual(tuple(gin.GINClassifier(5, 2)(x, adjacency).shape), (2,))
        isolate = torch.zeros(4, 4)
        attention = gat.GraphAttentionLayer(5, 3)
        first = attention(x, isolate)[0]
        changed = x.clone()
        changed[1:] += 100
        self.assertTrue(torch.allclose(first, attention(changed, isolate)[0], atol=1e-6))
        layer = gin.GINLayer(5)
        layer.mlp = nn.Identity()
        few = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        more = torch.tensor([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
        self.assertTrue(torch.allclose(layer(torch.ones(3, 5), more)[0]
                                       - layer(torch.ones(3, 5), few)[0], torch.ones(5)))

    def test_point_cloud_and_nerf(self):
        pointnet = source("PointNet", "pointnet.py")
        dgcnn = source("DGCNN", "dgcnn.py")
        nerf = source("NeRF", "nerf.py")
        points = torch.randn(2, 12, 3)
        permutation = torch.randperm(12)
        for model in (pointnet.PointNet(4), dgcnn.DGCNN(4, k=3)):
            model.eval()
            self.assertTrue(torch.allclose(model(points), model(points[:, permutation]), atol=1e-5))
        model = nerf.NeRF(hidden=32)
        origins = torch.zeros(2, 3)
        directions = torch.tensor([[0., 0., 1.], [0., 1., 0.]])
        depths = torch.linspace(.1, 1, 6).expand(2, 6)
        color, depth, opacity = nerf.render_rays(model, origins, directions, depths)
        self.assertEqual(tuple(color.shape), (2, 3))
        self.assertTrue(torch.isfinite(color).all() and torch.isfinite(depth).all())
        self.assertTrue(((opacity >= 0) & (opacity <= 1.00001)).all())

    def test_self_supervision(self):
        moco = source("MoCo", "moco.py")
        byol = source("BYOL", "byol.py")
        dino = source("DINO", "dino.py")
        mae = source("MAE", "mae.py")
        images = torch.randn(3, 3, 32, 32)
        contrast = moco.MoCo(dim=16, queue_size=7)
        contrast(images, images + .02).backward()
        self.assertEqual(int(contrast.queue_pointer), 3)
        self.assertTrue(all(p.grad is None for p in contrast.key_encoder.parameters()))
        bootstrap = byol.BYOL(dim=16, momentum=.5)
        bootstrap(images, images + .02).backward()
        self.assertTrue(all(p.grad is None for p in bootstrap.target_encoder.parameters()))
        teacher = dino.DINO(dim=32, prototypes=16)
        initial_center = teacher.center.clone()
        teacher(images, images + .02).backward()
        self.assertTrue(all(p.grad is None for p in teacher.teacher.parameters()))
        self.assertFalse(torch.equal(initial_center, teacher.center))
        predicted, masked, loss = mae.MaskedAutoencoder()(images)
        self.assertEqual(tuple(predicted.shape), (3, 16, 192))
        self.assertTrue((masked.sum(dim=1) == 12).all())
        self.assertTrue(torch.isfinite(loss))

    def test_blip_causal_and_losses(self):
        blip = source("BLIP", "blip.py")
        model = blip.BLIP(vocab_size=24, dim=32)
        images = torch.randn(3, 3, 32, 32)
        ids = torch.tensor([[1, 3, 4, 2], [1, 5, 6, 2], [1, 7, 8, 2]])
        losses = model(images, ids)
        for name in ("itc", "itm", "lm"):
            self.assertTrue(torch.isfinite(losses[name]))
        memory = model.visual(images)
        changed = ids.clone()
        changed[:, -1] = 9
        self.assertTrue(torch.allclose(model.caption_logits(memory, ids)[:, :-1],
                                       model.caption_logits(memory, changed)[:, :-1], atol=1e-5))
        captions, keep, scores = model.bootstrap_and_filter(images, length=4)
        self.assertEqual(tuple(captions.shape), (3, 4))
        self.assertEqual(tuple(keep.shape), (3,))
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())

    def test_rl_and_recommenders(self):
        ddpg = source("DDPG", "ddpg.py")
        a2c = source("A2C", "a2c.py")
        sac = source("SAC", "sac.py")
        ncf = source("NCF", "ncf.py")
        wide = source("Wide and Deep", "wide_and_deep.py")
        deepfm = source("DeepFM", "deepfm.py")
        self.assertTrue((ddpg.Actor(4, 2)(torch.randn(5, 4)).abs() <= 1).all())
        returns = a2c.n_step_returns(torch.tensor([[1.], [2.], [3.]]),
                                      torch.tensor([[0.], [1.], [0.]]),
                                      torch.tensor([4.]), gamma=.5)
        self.assertTrue(torch.allclose(returns[:, 0], torch.tensor([2., 2., 5.])))
        actions, log_prob = sac.GaussianActor(4, 2).sample(torch.randn(5, 4))
        self.assertTrue((actions.abs() < 1).all() and torch.isfinite(log_prob).all())
        self.assertEqual(tuple(ncf.NeuMF(8, 9)(torch.tensor([1, 2]),
                                               torch.tensor([3, 4])).shape), (2,))
        self.assertEqual(tuple(wide.WideAndDeep(3, 10, 6)(torch.randn(2, 3),
                         torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1], [2]])).shape), (2,))
        self.assertEqual(tuple(deepfm.DeepFM([5, 6])(torch.tensor([[1, 2], [3, 4]])).shape), (2,))


if __name__ == "__main__":
    unittest.main()
