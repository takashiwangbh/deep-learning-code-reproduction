"""Mask R-CNN：本地 Faster R-CNN 两阶段核心 + 对齐采样 + 实例掩码头。"""
import torch
from torch import nn
from torch.nn import functional as F
from faster_rcnn import FasterRCNN, box_iou


class MaskRCNN(FasterRCNN):
    def __init__(self, classes=3, width=32):
        super().__init__(classes, width)
        self.mask_head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(width, width, 2, stride=2), nn.ReLU(),
            nn.Conv2d(width, classes + 1, 1),
        )

    @staticmethod
    def _align(feature, boxes, size):
        """每个网格 bin 用双线性插值取样，避免 RoI Pool 的整数坐标量化。"""
        samples = []
        for box in boxes:
            x1, y1, x2, y2 = box.unbind()
            coordinate = (torch.arange(size, device=feature.device, dtype=feature.dtype) + .5) / size
            gx = x1 + coordinate * (x2 - x1)
            gy = y1 + coordinate * (y2 - y1)
            yy, xx = torch.meshgrid(gy, gx, indexing="ij")
            grid = torch.stack([2 * xx - 1, 2 * yy - 1], -1)[None]
            samples.append(F.grid_sample(feature, grid, mode="bilinear",
                                         padding_mode="zeros", align_corners=False))
        return torch.cat(samples, 0)

    def _roi_features(self, feature, boxes):
        return self._align(feature, boxes, 3)

    def forward(self, image, topk=12):
        predictions = super().forward(image, topk)
        for item in predictions:
            item["mask_logits"] = self.mask_head(item["roi_features"])
        return predictions

    def training_loss(self, image, gt_boxes, gt_labels, gt_masks):
        detection = super().training_loss(image, gt_boxes, gt_labels)
        predictions = self(image)
        mask_losses = []
        for item, boxes, labels, masks in zip(predictions, gt_boxes, gt_labels, gt_masks):
            if boxes.numel() == 0:
                mask_losses.append(item["mask_logits"].sum() * 0)
                continue
            roi_boxes = torch.cat([item["proposals"], boxes.detach()], 0)
            mask_logits = self.mask_head(self._roi_features(item["feature"], roi_boxes))
            overlap, matched = box_iou(roi_boxes, boxes).max(1)
            positive = torch.nonzero(overlap >= .5, as_tuple=False).flatten()
            if positive.numel() == 0:
                mask_losses.append(mask_logits.sum() * 0)
                continue
            chosen = roi_boxes[positive]
            classes = labels[matched[positive]]
            prediction = mask_logits[positive, classes]
            targets = []
            for proposal, gt_index in zip(chosen, matched[positive]):
                target = self._align(masks[gt_index:gt_index+1, None].float(), proposal[None],
                                     prediction.shape[-1])[0, 0]
                targets.append(target)
            mask_losses.append(F.binary_cross_entropy_with_logits(prediction, torch.stack(targets)))
        return detection + torch.stack(mask_losses).mean()


if __name__ == "__main__":
    model = MaskRCNN(classes=3, width=16)
    image = torch.randn(1, 3, 32, 32)
    box = torch.tensor([[.2, .2, .7, .7]])
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 6:23, 6:23] = 1
    result = model(image)[0]
    loss = model.training_loss(image, [box], [torch.tensor([1])], mask)
    loss.backward()
    print("Mask R-CNN", tuple(result["class_logits"].shape),
          tuple(result["mask_logits"].shape), "loss", round(loss.item(), 4))
