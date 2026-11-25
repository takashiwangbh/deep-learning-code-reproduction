"""RetinaNet：两级 FPN、共享密集预测子网及 focal loss。"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def box_iou(a, b):
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp_min(0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    return inter / (area_a[:, None] + area_b[None] - inter).clamp_min(1e-8)


def anchors_for_map(h, w, scale, device):
    ys, xs = torch.meshgrid((torch.arange(h, device=device) + .5) / h,
                            (torch.arange(w, device=device) + .5) / w, indexing="ij")
    center = torch.stack([xs, ys], -1)[:, :, None]
    wh = torch.tensor([(scale, scale), (scale * math.sqrt(2), scale / math.sqrt(2)),
                       (scale / math.sqrt(2), scale * math.sqrt(2))], device=device)
    return torch.cat([center - wh / 2, center + wh / 2], -1).reshape(-1, 4).clamp(0, 1)


def encode(boxes, anchors):
    bc, bw = (boxes[:, :2] + boxes[:, 2:]) / 2, (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-6)
    ac, aw = (anchors[:, :2] + anchors[:, 2:]) / 2, (anchors[:, 2:] - anchors[:, :2]).clamp_min(1e-6)
    return torch.cat([(bc - ac) / aw, torch.log(bw / aw)], -1)


def force_unique_matches(iou):
    """贪心挑唯一 anchor–GT 对，避免多个目标都被同一最佳 anchor 覆盖。"""
    anchors, targets = iou.shape
    if targets > anchors:
        raise ValueError("真实框数不能多于 anchor 数")
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


def focal_loss(logits, targets, valid=None, alpha=.25, gamma=2.):
    probability = logits.sigmoid()
    pt = torch.where(targets.bool(), probability, 1 - probability)
    alpha_t = torch.where(targets.bool(), alpha, 1 - alpha)
    loss = alpha_t * (1 - pt).pow(gamma) * F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if valid is not None:
        loss = loss * valid[..., None]
    return loss.sum()


class RetinaNet(nn.Module):
    def __init__(self, classes=3, width=32):
        super().__init__()
        self.classes = classes
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU())
        self.down = nn.Sequential(nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.ReLU())
        self.lateral3 = nn.Conv2d(width, width, 1)
        self.lateral4 = nn.Conv2d(2 * width, width, 1)
        self.cls_tower = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                                       nn.Conv2d(width, 3 * classes, 3, padding=1))
        self.box_tower = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                                       nn.Conv2d(width, 3 * 4, 3, padding=1))
        nn.init.constant_(self.cls_tower[-1].bias, -math.log((1 - .01) / .01))

    def forward(self, image):
        c3 = self.stem(image)
        c4 = self.down(c3)
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        classes, offsets, anchors = [], [], []
        for feature, scale in ((p3, .18), (p4, .35)):
            b, _, h, w = feature.shape
            classes.append(self.cls_tower(feature).permute(0, 2, 3, 1).reshape(b, -1, self.classes))
            offsets.append(self.box_tower(feature).permute(0, 2, 3, 1).reshape(b, -1, 4))
            anchors.append(anchors_for_map(h, w, scale, feature.device))
        return torch.cat(classes, 1), torch.cat(offsets, 1), torch.cat(anchors)

    def training_loss(self, image, gt_boxes, gt_labels):
        logits, offsets, anchors = self(image)
        losses = []
        for score, delta, boxes, labels in zip(logits, offsets, gt_boxes, gt_labels):
            if boxes.numel() == 0:
                losses.append(focal_loss(score, torch.zeros_like(score)) + delta.sum() * 0)
                continue
            iou = box_iou(anchors, boxes)
            highest, matched = iou.max(1)
            positive = highest >= .5
            forced_anchor, forced_gt = force_unique_matches(iou)
            positive[forced_anchor] = True
            matched[forced_anchor] = forced_gt
            valid = (highest < .4) | positive
            target = torch.zeros_like(score)
            target[positive, labels[matched[positive]] - 1] = 1
            count = positive.sum().clamp_min(1)
            cls_loss = focal_loss(score, target, valid) / count
            box_loss = F.smooth_l1_loss(delta[positive], encode(boxes[matched[positive]], anchors[positive]))
            losses.append(cls_loss + box_loss)
        return torch.stack(losses).mean()


if __name__ == "__main__":
    model = RetinaNet(classes=3, width=16)
    image = torch.randn(1, 3, 64, 64)
    scores, offsets, anchors = model(image)
    loss = model.training_loss(image, [torch.tensor([[.2, .2, .7, .7]])], [torch.tensor([2])])
    loss.backward()
    print("RetinaNet", tuple(scores.shape), tuple(offsets.shape), tuple(anchors.shape), "loss", round(loss.item(), 4))
