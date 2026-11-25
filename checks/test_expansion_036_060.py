"""036–060 新项目的 CPU 形状、梯度和机制检查。仓库根目录可直接运行。"""
import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def load(folder, filename):
    path = ROOT / folder / filename
    spec = importlib.util.spec_from_file_location(folder.replace(" ", "_").replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class ExpansionChecks(unittest.TestCase):
    def test_language_models(self):
        t5 = load("T5", "t5.py")
        model = t5.T5(vocab=24, dim=16, heads=4, layers=1).eval()
        source = torch.tensor([[4, 5, 0]])
        a = model(source, torch.tensor([[1, 2, 3]]))
        b = model(source, torch.tensor([[1, 2, 9]]))
        torch.testing.assert_close(a[:, :2], b[:, :2])  # 解码不能偷看未来
        self.assertEqual(a.shape, (1, 3, 24))
        self.assertTrue(torch.isfinite(a).all())

        xl = load("Transformer-XL", "transformer_xl.py")
        network = xl.TransformerXL(vocab=24, dim=16, heads=4, layers=1, mem_len=3)
        _, memory = network(torch.tensor([[1, 2, 3, 4]]))
        logits, updated = network(torch.tensor([[5, 6]]), memory)
        self.assertEqual(logits.shape, (1, 2, 24))
        self.assertEqual(updated[0].shape, (1, 3, 16))
        self.assertFalse(updated[0].requires_grad)  # 不跨段反传

        longformer = load("Longformer", "longformer.py")
        attention = longformer.LocalGlobalAttention(dim=8, heads=2, window=1).eval()
        x = torch.randn(1, 5, 8)
        changed = x.clone()
        changed[:, 4] += 10
        no_global = torch.zeros(5, dtype=torch.bool)
        torch.testing.assert_close(attention(x, no_global)[:, 0], attention(changed, no_global)[:, 0])
        global_end = no_global.clone()
        global_end[4] = True
        self.assertFalse(torch.allclose(attention(x, global_end)[:, 0], attention(changed, global_end)[:, 0]))

        lin = load("Linformer", "linformer.py")
        self.assertEqual(lin.Linformer(dim=16, heads=4, max_length=8, rank=3)(
            torch.randint(0, 128, (2, 6))).shape, (2, 6, 128))
        performer = load("Performer", "performer.py")
        fast = performer.Performer(vocab=24, dim=16, heads=4, features=8).eval()
        a = fast(torch.tensor([[1, 2, 3, 4]]))
        b = fast(torch.tensor([[1, 2, 3, 9]]))
        torch.testing.assert_close(a[:, :3], b[:, :3], rtol=1e-5, atol=1e-5)
        extreme = performer.FavorAttention(dim=1, heads=1, features=1)
        with torch.no_grad():
            extreme.qkv.weight.fill_(1)
            extreme.out.weight.fill_(1)
            extreme.omega.fill_(100)
        early = extreme(torch.tensor([[[1.], [0.]]]))
        changed_future = extreme(torch.tensor([[[1.], [100.]]]))
        torch.testing.assert_close(early[:, 0], changed_future[:, 0])
        torch.testing.assert_close(early[:, 0], torch.ones_like(early[:, 0]))
        switch = load("Switch Transformer", "switch_transformer.py")
        moe = switch.SwitchFFN(dim=8, experts=2, capacity_factor=1)
        with torch.no_grad():
            moe.router.weight.zero_()
        dispatched, aux = moe(torch.ones(1, 4, 8))
        self.assertTrue(torch.equal(dispatched[:, 2:], torch.zeros_like(dispatched[:, 2:])))
        self.assertTrue(torch.isfinite(aux))

    def test_vision_transformers(self):
        deit = load("DeiT", "deit.py")
        cls, dist = deit.DeiT(image_size=16, patch=8, dim=16, heads=4, depth=1, classes=3)(
            torch.randn(2, 3, 16, 16))
        self.assertEqual((cls.shape, dist.shape), ((2, 3), (2, 3)))
        loss = deit.hard_distillation_loss(cls, dist, torch.tensor([0, 1]), torch.randn(2, 3))
        loss.backward()

        swin = load("Swin Transformer", "swin.py")
        x = torch.randn(1, 8, 8, 4)
        torch.testing.assert_close(swin.reverse(swin.partition(x, 4), 8, 8, 4), x)
        self.assertTrue(swin.shifted_mask(8, 8, 4, 2, x.device).any())
        self.assertEqual(swin.SwinTransformer(dim=16)(torch.randn(1, 3, 32, 32)).shape, (1, 10))
        shifted = swin.SwinBlock(8, 2, 4, shift=2)
        unshifted = swin.SwinBlock(8, 2, 4, shift=0)
        unshifted.load_state_dict(shifted.state_dict())
        single_window = torch.randn(1, 4, 4, 8)
        torch.testing.assert_close(shifted(single_window), unshifted(single_window))

        pvt = load("PVT", "pvt.py")
        pyramid = pvt.PVT(dim=8).forward_features(torch.randn(1, 3, 64, 64))
        self.assertEqual([f.shape[-1] for f in pyramid], [16, 8, 4, 2])
        mixer = load("MLP-Mixer", "mlp_mixer.py")
        self.assertEqual(mixer.MLPMixer(image_size=16, patch=8, dim=16, depth=1)(
            torch.randn(2, 3, 16, 16)).shape, (2, 10))
        perceiver = load("Perceiver", "perceiver.py")
        net = perceiver.Perceiver(input_dim=3, dim=16, latents=4, depth=1)
        self.assertEqual(net(torch.randn(2, 5, 3)).shape, net(torch.randn(2, 8, 3)).shape)

    def test_segmentation(self):
        for folder, file, name in [
            ("FCN", "fcn.py", "FCN8s"),
            ("SegNet", "segnet.py", "SegNet"),
            ("DeepLab V3+", "deeplab_v3plus.py", "DeepLabV3Plus"),
        ]:
            with self.subTest(folder=folder):
                model = getattr(load(folder, file), name)(classes=3, width=8)
                image = torch.randn(1, 3, 64, 64)
                y = model(image)
                self.assertEqual(y.shape, (1, 3, 64, 64))
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(y).all())

    def test_detection(self):
        box = [torch.tensor([[.2, .2, .7, .7]])]
        label = [torch.tensor([1])]
        image = torch.randn(1, 3, 32, 32)
        for folder, file, name in [
            ("Faster R-CNN", "faster_rcnn.py", "FasterRCNN"),
            ("SSD", "ssd.py", "SSD"),
            ("RetinaNet", "retinanet.py", "RetinaNet"),
        ]:
            with self.subTest(folder=folder):
                model = getattr(load(folder, file), name)(classes=3, width=8)
                loss = model.training_loss(image, box, label)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters()))

        detr = load("DETR", "detr.py")
        rows, cols = detr.optimal_assignment(torch.tensor([[1., 2.], [2., 100.], [100., 1.]]))
        self.assertEqual(rows.tolist(), [0, 2])
        self.assertEqual(cols.tolist(), [0, 1])
        model = detr.DETR(classes=3, dim=16, queries=5)
        det_loss = model.set_loss(image, [torch.tensor([[.5, .5, .3, .3]])], [torch.tensor([1])])
        self.assertTrue(torch.isfinite(det_loss))
        det_loss.backward()

        mask = load("Mask R-CNN", "mask_rcnn.py")
        model = mask.MaskRCNN(classes=3, width=8)
        gt_mask = torch.zeros(1, 1, 32, 32)
        gt_mask[:, :, 6:23, 6:23] = 1
        mask_loss = model.training_loss(image, box, label, gt_mask)
        self.assertTrue(torch.isfinite(mask_loss))
        mask_loss.backward()
        self.assertGreater(model.mask_head[-1].weight.grad.abs().sum().item(), 0)

    def test_matching_collisions_and_empty_images(self):
        # 两个 GT 都把 anchor 0 选为首选，仍应得到两个不同的正 anchor。
        overlap = torch.tensor([[.9, .85], [.8, .1], [.1, .7]])
        for folder, file in [
            ("Faster R-CNN", "faster_rcnn.py"),
            ("SSD", "ssd.py"),
            ("RetinaNet", "retinanet.py"),
        ]:
            with self.subTest(folder=folder):
                module = load(folder, file)
                anchors, targets = module.force_unique_matches(overlap)
                self.assertEqual(len(anchors.unique()), 2)
                self.assertEqual(set(targets.tolist()), {0, 1})

        image = torch.randn(1, 3, 32, 32)
        empty_boxes = [torch.empty(0, 4)]
        empty_labels = [torch.empty(0, dtype=torch.long)]
        for folder, file, name in [
            ("Faster R-CNN", "faster_rcnn.py", "FasterRCNN"),
            ("SSD", "ssd.py", "SSD"),
            ("RetinaNet", "retinanet.py", "RetinaNet"),
        ]:
            with self.subTest(empty=folder):
                model = getattr(load(folder, file), name)(classes=3, width=8)
                loss = model.training_loss(image, empty_boxes, empty_labels)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
        detr = load("DETR", "detr.py")
        self.assertTrue(torch.isfinite(detr.DETR(classes=3, dim=16, queries=5).set_loss(
            image, empty_boxes, empty_labels)))
        mask = load("Mask R-CNN", "mask_rcnn.py")
        self.assertTrue(torch.isfinite(mask.MaskRCNN(classes=3, width=8).training_loss(
            image, empty_boxes, empty_labels, torch.empty(1, 0, 32, 32))))


if __name__ == "__main__":
    unittest.main()
