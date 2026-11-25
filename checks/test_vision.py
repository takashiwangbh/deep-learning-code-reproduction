"""无需下载数据：检查独立视觉项目的形状、梯度及关键公式。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for project in ("LeNet-5", "AlexNet", "VGG", "GoogLeNet", "DenseNet",
                "MobileNetV2", "U-Net", "Vision Transformer", "YOLOv1", "SimCLR"):
    sys.path.insert(0, str(ROOT / project))

import torch

from alexnet import AlexNet
from densenet121 import DenseNet121
from googlenet import GoogLeNet
from lenet5 import LeNet5
from mobilenet_v2 import MobileNetV2
from simclr import SimCLR, make_two_views, nt_xent_loss
from unet import UNet
from vgg16 import VGG16
from vit import VisionTransformer
from yolov1 import YOLOv1, YOLOv1Loss, box_iou, responsible_box_mask


def check(name: str, model: torch.nn.Module, shape: tuple[int, ...],
          expected: tuple[int, ...]) -> None:
    x = torch.randn(*shape)
    result = model(x)
    outputs = result if isinstance(result, tuple) else (result,)
    assert outputs[0].shape == expected, (name, outputs[0].shape, expected)
    assert all(output.shape == expected for output in outputs), name
    sum(output.square().mean() for output in outputs).backward()
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all()
                             for g in gradients), f"{name}: 梯度缺失或非有限值"
    print(f"{name:14s} {shape} -> {expected}; backward OK")


def check_yolov1() -> None:
    # 一格两个候选框：小框 IoU=1/4，大框 IoU=1，第二个应负责。
    candidates = torch.tensor([[[[[0.5, 0.5, 0.2, 0.2],
                                  [0.5, 0.5, 0.4, 0.4]]]]])
    truth = torch.tensor([[[[0.5, 0.5, 0.4, 0.4]]]])
    torch.testing.assert_close(box_iou(candidates, truth, 1),
                               torch.tensor([[[[0.25, 1.0]]]]))
    chosen = responsible_box_mask(candidates, truth, torch.ones(1, 1, 1), 1)
    assert chosen.tolist() == [[[[False, True]]]]

    # 完全匹配的责任框，confidence=0：只有 IoU 置信度项为 1。
    prediction = torch.tensor([[[[1., 0.5, 0.5, 0.2, 0.2, 0.,
                                  0.5, 0.5, 0.4, 0.4, 0.]]]], requires_grad=True)
    loss = YOLOv1Loss(1, 2, 1)(prediction, torch.ones(1, 1, 1),
                                 truth, torch.ones(1, 1, 1, 1))
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    torch.testing.assert_close(prediction.grad[..., 10], torch.tensor([[[-2.0]]]))
    torch.testing.assert_close(prediction.grad[..., 1:5], torch.zeros(1, 1, 1, 4))

    # 无物体的网格只惩罚两个框的置信度，不回归坐标或类别。
    empty_prediction = torch.tensor([[[[9., 0.1, 0.2, 0.3, 0.4, 0.5,
                                        0.5, 0.6, 0.7, 0.8, -0.5]]]],
                                    requires_grad=True)
    empty_loss = YOLOv1Loss(1, 2, 1)(empty_prediction,
                                       torch.zeros(1, 1, 1),
                                       torch.zeros(1, 1, 1, 4),
                                       torch.zeros(1, 1, 1, 1))
    torch.testing.assert_close(empty_loss, torch.tensor(0.25))
    empty_loss.backward()
    assert torch.count_nonzero(empty_prediction.grad[..., [0, 1, 2, 3, 4, 6, 7, 8, 9]]) == 0

    model = YOLOv1(2, 2, 3, width_mult=0.125, hidden=64)
    assert sum(isinstance(layer, torch.nn.Conv2d) for layer in model.modules()) == 24
    outputs = model(torch.randn(2, 3, 128, 128))
    assert outputs.shape == (2, 2, 2, 13)
    objects = torch.zeros(2, 2, 2)
    objects[:, 0, 1] = 1
    boxes = torch.zeros(2, 2, 2, 4)
    boxes[:, 0, 1] = torch.tensor([0.5, 0.5, 0.25, 0.4])
    classes = torch.zeros(2, 2, 2, 3)
    classes[:, 0, 1, 2] = 1
    train_loss = YOLOv1Loss(2, 2, 3)(outputs, objects, boxes, classes)
    train_loss.backward()
    grad = model.features[0][0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    print("YOLOv1         IoU / responsible box / five losses / 24-conv backward OK")


def check_simclr() -> None:
    # N=2、单位正交向量：每个锚点的分母是 exp(1)+2，绝不含自身的 exp(1)。
    basis = torch.eye(2)
    exact = nt_xent_loss(basis, basis, temperature=1.0)
    expected = torch.log1p(2 * torch.exp(torch.tensor(-1.0)))
    torch.testing.assert_close(exact, expected)
    assert nt_xent_loss(basis, basis.flip(0), temperature=1.0) > exact

    images = torch.rand(2, 3, 32, 32)
    view1, view2 = make_two_views(images)
    assert view1.shape == view2.shape == images.shape
    model = SimCLR(width=8, projection_hidden=32, projection_dim=16)
    z1, z2 = model(view1, view2)
    assert z1.shape == z2.shape == (2, 16)
    loss = nt_xent_loss(z1, z2)
    loss.backward()
    for parameter in (model.encoder.stem[0].weight, model.projector[0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert model.representation(images).shape == (2, 64)
    print("SimCLR         two views / NT-Xent denominator / encoder backward OK")


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(2)
    check("LeNet-5", LeNet5(10), (2, 1, 32, 32), (2, 10))
    check("AlexNet", AlexNet(10, 0.25, 32, 1), (2, 3, 96, 96), (2, 10))
    check("VGG-16", VGG16(10, 0.125, 32, 1), (2, 3, 64, 64), (2, 10))
    check("GoogLeNet", GoogLeNet(10, 0.125, aux_logits=True),
          (2, 3, 64, 64), (2, 10))
    check("DenseNet-121", DenseNet121(10, 4, (1, 1, 1, 1), 8),
          (2, 3, 64, 64), (2, 10))
    check("MobileNetV2", MobileNetV2(10, 0.25, 32),
          (2, 3, 64, 64), (2, 10))
    check("U-Net", UNet(3, 2, 8), (2, 3, 65, 67), (2, 2, 65, 67))
    check("ViT", VisionTransformer(32, 8, 10, 32, 2, 4),
          (2, 3, 32, 32), (2, 10))
    check_yolov1()
    check_simclr()


if __name__ == "__main__":
    main()
