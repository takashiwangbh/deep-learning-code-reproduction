"""SSD：多尺度特征图、默认框、匹配与困难负例挖掘。"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def box_iou(a, b):
    left = torch.maximum(a[:, None, :2], b[None, :, :2])
    right = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (right - left).clamp_min(0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    return inter / (area_a[:, None] + area_b[None] - inter).clamp_min(1e-8)


def default_boxes(h, w, scale, device):
    ys, xs = torch.meshgrid((torch.arange(h, device=device) + .5) / h,
                            (torch.arange(w, device=device) + .5) / w, indexing="ij")
    centers = torch.stack([xs, ys], -1)[:, :, None, :]
    sizes = torch.tensor([
        (scale, scale), (scale * math.sqrt(2), scale / math.sqrt(2)),
        (scale / math.sqrt(2), scale * math.sqrt(2)), (scale * 1.25, scale * 1.25)
    ], device=device)
    return torch.cat([centers - sizes / 2, centers + sizes / 2], -1).reshape(-1, 4).clamp(0, 1)


def encode(boxes, anchors):
    bc, bw = (boxes[:, :2] + boxes[:, 2:]) / 2, (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-6)
    ac, aw = (anchors[:, :2] + anchors[:, 2:]) / 2, (anchors[:, 2:] - anchors[:, :2]).clamp_min(1e-6)
    return torch.cat([10 * (bc - ac) / aw, 5 * torch.log(bw / aw)], -1)


def force_unique_matches(iou):
    """按 IoU 从高到低挑互不冲突的 anchor–GT 对，避免一个 anchor 覆盖多个 GT。"""
    anchors, targets = iou.shape
    if targets > anchors:
        raise ValueError("真实框数不能多于默认框数")
    used_anchor = torch.zeros(anchors, dtype=torch.bool, device=iou.device)
    used_target = torch.zeros(targets, dtype=torch.bool, device=iou.device)
    chosen_anchor, chosen_target = [], []
    for flat in iou.flatten().argsort(descending=True).tolist():
        anchor, target = divmod(flat, targets)
        if not used_anchor[anchor] and not used_target[target]:
            chosen_anchor.append(anchor)
            chosen_target.append(target)
            used_anchor[anchor] = True
            used_target[target] = True
            if len(chosen_target) == targets:
                break
    return torch.tensor(chosen_anchor, dtype=torch.long, device=iou.device), torch.tensor(
        chosen_target, dtype=torch.long, device=iou.device)


class SSD(nn.Module):
    def __init__(self, classes=3, width=16):
        super().__init__()
        self.classes = classes
        self.backbone = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.extra = nn.Sequential(nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.ReLU())
        self.cls_heads = nn.ModuleList([nn.Conv2d(width, 4 * (classes + 1), 3, padding=1),
                                         nn.Conv2d(2 * width, 4 * (classes + 1), 3, padding=1)])
        self.box_heads = nn.ModuleList([nn.Conv2d(width, 4 * 4, 3, padding=1),
                                         nn.Conv2d(2 * width, 4 * 4, 3, padding=1)])

    def forward(self, image):
        f1 = self.backbone(image)
        f2 = self.extra(f1)
        class_scores, box_offsets, anchors = [], [], []
        for feat, cls, box, scale in zip((f1, f2), self.cls_heads, self.box_heads, (.15, .35)):
            b, _, h, w = feat.shape
            class_scores.append(cls(feat).permute(0, 2, 3, 1).reshape(b, -1, self.classes + 1))
            box_offsets.append(box(feat).permute(0, 2, 3, 1).reshape(b, -1, 4))
            anchors.append(default_boxes(h, w, scale, feat.device))
        return torch.cat(class_scores, 1), torch.cat(box_offsets, 1), torch.cat(anchors, 0)

    def training_loss(self, image, gt_boxes, gt_labels):
        logits, offsets, anchors = self(image)
        losses = []
        for score, delta, boxes, labels in zip(logits, offsets, gt_boxes, gt_labels):
            if boxes.numel() == 0:
                per_anchor = F.cross_entropy(
                    score, torch.zeros(len(anchors), dtype=torch.long, device=score.device),
                    reduction="none",
                )
                losses.append(per_anchor.topk(min(3, len(anchors))).values.mean() + delta.sum() * 0)
                continue
            iou = box_iou(anchors, boxes)
            best_iou, best_gt = iou.max(1)
            positive = best_iou >= .5
            forced_anchor, forced_gt = force_unique_matches(iou)
            positive[forced_anchor] = True
            best_gt[forced_anchor] = forced_gt
            target = torch.zeros(len(anchors), dtype=torch.long, device=anchors.device)
            target[positive] = labels[best_gt[positive]]
            per_anchor = F.cross_entropy(score, target, reduction="none")
            neg_loss = per_anchor.masked_fill(positive, -1)
            negatives = min(int(positive.sum()) * 3, int((~positive).sum()))
            selected = positive.clone()
            if negatives:
                selected[neg_loss.topk(negatives).indices] = True
            cls_loss = per_anchor[selected].sum() / positive.sum().clamp_min(1)
            box_loss = F.smooth_l1_loss(delta[positive], encode(boxes[best_gt[positive]], anchors[positive]))
            losses.append(cls_loss + box_loss)
        return torch.stack(losses).mean()


if __name__ == "__main__":
    model = SSD(classes=3)
    image = torch.randn(1, 3, 64, 64)
    scores, offsets, anchors = model(image)
    loss = model.training_loss(image, [torch.tensor([[.2, .2, .7, .7]])], [torch.tensor([2])])
    loss.backward()
    print("SSD", tuple(scores.shape), tuple(offsets.shape), tuple(anchors.shape), "loss", round(loss.item(), 4))
