"""YOLOv1：24 层卷积检测器、网格预测和论文中的五项平方误差损失。

论文：https://arxiv.org/abs/1506.02640
预测布局 [N, S, S, C + B*5]：先 C 个条件类别分数，再 B 组 (x,y,w,h,conf)。
x/y 是相对于所在网格的中心偏移；w/h 是相对于整张图的宽高。
"""

import torch
import torch.nn.functional as F
from torch import nn


class ConvLeaky(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel: int,
                 stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel, stride=stride,
                      padding=kernel // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )


class YOLOv1(nn.Module):
    """默认 VOC 配置 S=7、B=2、C=20；`width_mult` 可缩小 CPU 示例。

主干保留论文中的 24 个卷积、交替的 1×1 降维和两层全连接。
只有 w/h 经过 softplus 保证正值，以免训练初期 sqrt(w/h) 出现 NaN；
其余输出仍是线性分数。默认全尺寸模型很大，请先运行底部小模型示例。
"""

    def __init__(self, grid_size: int = 7, num_boxes: int = 2,
                 num_classes: int = 20, width_mult: float = 1.0,
                 hidden: int = 4096):
        super().__init__()
        if min(grid_size, num_boxes, num_classes, hidden) <= 0 or width_mult <= 0:
            raise ValueError("网格、框数、类别、隐藏层和宽度比例必须为正")
        self.grid_size, self.num_boxes, self.num_classes = grid_size, num_boxes, num_classes
        layers: list[nn.Module] = []
        in_channels = 3

        def add(channels: int, kernel: int, stride: int = 1) -> None:
            nonlocal in_channels
            out_channels = max(4, round(channels * width_mult))
            layers.append(ConvLeaky(in_channels, out_channels, kernel, stride))
            in_channels = out_channels

        add(64, 7, 2)
        layers.append(nn.MaxPool2d(2, 2))
        add(192, 3)
        layers.append(nn.MaxPool2d(2, 2))
        for channels, kernel in ((128, 1), (256, 3), (256, 1), (512, 3)):
            add(channels, kernel)
        layers.append(nn.MaxPool2d(2, 2))
        for _ in range(4):
            add(256, 1)
            add(512, 3)
        add(512, 1)
        add(1024, 3)
        layers.append(nn.MaxPool2d(2, 2))
        for _ in range(2):
            add(512, 1)
            add(1024, 3)
        add(1024, 3)
        add(1024, 3, 2)
        add(1024, 3)
        add(1024, 3)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(in_channels * grid_size * grid_size, hidden),
            nn.LeakyReLU(0.1, inplace=True), nn.Dropout(0.5),
            nn.Linear(hidden, grid_size * grid_size * (num_classes + 5 * num_boxes)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        raw = self.head(self.pool(self.features(images)))
        raw = raw.reshape(batch, self.grid_size, self.grid_size, -1)
        classes = raw[..., :self.num_classes]
        boxes = raw[..., self.num_classes:].reshape(
            batch, self.grid_size, self.grid_size, self.num_boxes, 5
        )
        # 模型尺寸必须为正；原论文的最后一层完全线性，这里只对宽高加约束。
        boxes = torch.cat((boxes[..., :2], F.softplus(boxes[..., 2:4]),
                           boxes[..., 4:5]), dim=-1)
        return torch.cat((classes, boxes.flatten(start_dim=-2)), dim=-1)


def box_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor,
            grid_size: int, eps: float = 1e-8) -> torch.Tensor:
    """比较 [N,S,S,B,4] 预测框与 [N,S,S,4] 真值，返回 [N,S,S,B] IoU。"""
    if pred_boxes.ndim != 5 or target_boxes.ndim != 4 or grid_size <= 0:
        raise ValueError("框张量形状或 grid_size 无效")
    if pred_boxes.shape[:3] != target_boxes.shape[:3] or pred_boxes.shape[-1] != 4:
        raise ValueError("预测框与真值框的网格或坐标维度不匹配")
    if pred_boxes.shape[1:3] != (grid_size, grid_size) or target_boxes.shape[-1] != 4:
        raise ValueError("网格尺寸必须与 grid_size 匹配")

    row, col = torch.meshgrid(
        torch.arange(grid_size, device=pred_boxes.device),
        torch.arange(grid_size, device=pred_boxes.device), indexing="ij"
    )
    cell_offset = torch.stack((col, row), dim=-1).to(pred_boxes.dtype)[None, :, :, None]

    def corners(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        center = (boxes[..., :2] + cell_offset) / grid_size
        half_size = boxes[..., 2:4].clamp_min(0) / 2
        return center - half_size, center + half_size

    pred_min, pred_max = corners(pred_boxes)
    truth_min, truth_max = corners(target_boxes.unsqueeze(-2))
    intersection_size = (torch.minimum(pred_max, truth_max)
                         - torch.maximum(pred_min, truth_min)).clamp_min(0)
    intersection = intersection_size.prod(dim=-1)
    pred_area = pred_boxes[..., 2:4].clamp_min(0).prod(dim=-1)
    truth_area = target_boxes[..., 2:4].clamp_min(0).prod(dim=-1).unsqueeze(-1)
    union = pred_area + truth_area - intersection
    return intersection / union.clamp_min(eps)


def responsible_box_mask(pred_boxes: torch.Tensor, target_boxes: torch.Tensor,
                         object_mask: torch.Tensor, grid_size: int) -> torch.Tensor:
    """每个有物体的网格只把当前 IoU 最大的预测框设为责任框。"""
    ious = box_iou(pred_boxes, target_boxes, grid_size)
    best = ious.detach().argmax(dim=-1)
    one_hot = F.one_hot(best, num_classes=pred_boxes.shape[-2]).bool()
    return one_hot & object_mask.bool().unsqueeze(-1)


class YOLOv1Loss(nn.Module):
    """坐标 + 有物体框置信度 + 无责任框置信度 + 类别，按 batch 求均值。

    object_mask: [N,S,S]；target_boxes: [N,S,S,4]；
    target_classes: [N,S,S,C] one-hot。每格至多一个真值物体。
    """

    def __init__(self, grid_size: int = 7, num_boxes: int = 2,
                 num_classes: int = 20, lambda_coord: float = 5.0,
                 lambda_noobj: float = 0.5):
        super().__init__()
        if min(grid_size, num_boxes, num_classes) <= 0 or lambda_coord < 0 or lambda_noobj < 0:
            raise ValueError("网格、框数、类别和损失权重无效")
        self.grid_size, self.num_boxes, self.num_classes = grid_size, num_boxes, num_classes
        self.lambda_coord, self.lambda_noobj = lambda_coord, lambda_noobj

    def forward(self, predictions: torch.Tensor, object_mask: torch.Tensor,
                target_boxes: torch.Tensor, target_classes: torch.Tensor) -> torch.Tensor:
        n, s, _, channels = predictions.shape
        expected = (n, self.grid_size, self.grid_size)
        if (s != self.grid_size or predictions.shape[2] != s or
                channels != self.num_classes + 5 * self.num_boxes or
                object_mask.shape != expected or
                target_boxes.shape != (*expected, 4) or
                target_classes.shape != (*expected, self.num_classes)):
            raise ValueError("YOLOv1 预测或目标张量形状不匹配")
        pred_classes = predictions[..., :self.num_classes]
        pred_boxes = predictions[..., self.num_classes:].reshape(
            n, s, s, self.num_boxes, 5
        )
        coords, confidence = pred_boxes[..., :4], pred_boxes[..., 4]
        responsible = responsible_box_mask(coords, target_boxes, object_mask, s)
        ious = box_iou(coords, target_boxes, s).detach()
        truth = target_boxes.unsqueeze(-2)

        # 仅责任框回归 x/y 与 sqrt(w/h)。模型 forward 保证预测宽高 > 0。
        xy_error = (coords[..., :2] - truth[..., :2]).square().sum(dim=-1)
        wh_error = (coords[..., 2:4].clamp_min(1e-8).sqrt()
                    - truth[..., 2:4].clamp_min(0).sqrt()).square().sum(dim=-1)
        coordinate_loss = self.lambda_coord * ((xy_error + wh_error) * responsible).sum()

        # IoU 是当前预测框的训练目标；detach 防止把目标本身当作梯度路径。
        confidence_loss = ((confidence - ious).square() * responsible).sum()
        # 物体网格中未被选为责任框的其他框，置信度目标也为 0。
        no_object_loss = self.lambda_noobj * (
            confidence.square() * (~responsible)
        ).sum()
        class_loss = ((pred_classes - target_classes).square().sum(dim=-1)
                      * object_mask).sum()
        return (coordinate_loss + confidence_loss + no_object_loss + class_loss) / n


if __name__ == "__main__":
    torch.manual_seed(0)
    model = YOLOv1(grid_size=2, num_boxes=2, num_classes=3,
                   width_mult=0.125, hidden=64)
    image = torch.randn(2, 3, 128, 128)
    obj = torch.zeros(2, 2, 2, dtype=torch.bool)
    obj[:, 0, 1] = True
    boxes = torch.zeros(2, 2, 2, 4)
    boxes[:, 0, 1] = torch.tensor([0.5, 0.5, 0.25, 0.4])
    classes = torch.zeros(2, 2, 2, 3)
    classes[:, 0, 1, 2] = 1
    prediction = model(image)
    loss = YOLOv1Loss(2, 2, 3)(prediction, obj, boxes, classes)
    loss.backward()
    print("prediction:", tuple(prediction.shape), "loss:", round(loss.item(), 4))
