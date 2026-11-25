"""教学版 Faster R-CNN：共享特征、RPN、NMS、RoI 池化和两阶段损失。"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def box_iou(a, b):
    left = torch.maximum(a[:, None, :2], b[None, :, :2])
    right = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = (right - left).clamp_min(0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    return intersection / (area_a[:, None] + area_b[None] - intersection).clamp_min(1e-8)


def encode_boxes(boxes, anchors):
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-6)
    a_center = (anchors[:, :2] + anchors[:, 2:]) / 2
    a_size = (anchors[:, 2:] - anchors[:, :2]).clamp_min(1e-6)
    return torch.cat([(center - a_center) / a_size, (size / a_size).log()], -1)


def decode_boxes(delta, anchors):
    a_center = (anchors[:, :2] + anchors[:, 2:]) / 2
    a_size = (anchors[:, 2:] - anchors[:, :2]).clamp_min(1e-6)
    center = a_center + delta[:, :2] * a_size
    size = a_size * delta[:, 2:].clamp(-4, 4).exp()
    return torch.cat([center - size / 2, center + size / 2], -1).clamp(0, 1)


def nms(boxes, scores, threshold=0.7, limit=16):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() and len(keep) < limit:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        remaining = order[1:]
        order = remaining[box_iou(boxes[i:i+1], boxes[remaining])[0] <= threshold]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)


def make_anchors(h, w, device):
    ys, xs = torch.meshgrid(
        (torch.arange(h, device=device) + .5) / h,
        (torch.arange(w, device=device) + .5) / w, indexing="ij"
    )
    center = torch.stack([xs, ys], -1).reshape(-1, 2)
    sizes = []
    for scale in (.20, .35):
        for ratio in (.5, 1., 2.):
            sizes.append((scale * math.sqrt(ratio), scale / math.sqrt(ratio)))
    wh = torch.tensor(sizes, device=device, dtype=torch.float32)
    center = center[:, None, :].expand(-1, len(sizes), -1)
    wh = wh[None].expand(center.size(0), -1, -1)
    return torch.cat([center - wh / 2, center + wh / 2], -1).reshape(-1, 4)


def force_unique_matches(iou):
    """每个真实框选到不同 anchor，处理多个 GT 共用最佳 anchor 的冲突。"""
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


class FasterRCNN(nn.Module):
    def __init__(self, classes=3, width=32):
        super().__init__()
        self.classes = classes
        self.backbone = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.rpn_shared = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        self.rpn_objectness = nn.Conv2d(width, 6, 1)
        self.rpn_delta = nn.Conv2d(width, 6 * 4, 1)
        self.roi_head = nn.Sequential(nn.Flatten(), nn.Linear(width * 3 * 3, width * 2), nn.ReLU())
        self.class_head = nn.Linear(width * 2, classes + 1)  # 0 为背景
        self.box_head = nn.Linear(width * 2, 4)  # 教学版类别无关回归

    def _roi_features(self, feature, boxes):
        """教学版 RoI pooling；坐标离散化，Mask R-CNN 会替换为 RoIAlign。"""
        _, _, h, w = feature.shape
        crops = []
        for box in boxes:
            x1 = int(torch.floor(box[0] * w).clamp(0, w - 1).item())
            y1 = int(torch.floor(box[1] * h).clamp(0, h - 1).item())
            x2 = int(torch.ceil(box[2] * w).clamp(x1 + 1, w).item())
            y2 = int(torch.ceil(box[3] * h).clamp(y1 + 1, h).item())
            crops.append(F.adaptive_max_pool2d(feature[:, :, y1:y2, x1:x2], (3, 3)))
        return torch.cat(crops, 0)

    def forward(self, image, topk=12):
        feature = self.backbone(image)
        b, _, h, w = feature.shape
        anchors = make_anchors(h, w, feature.device)
        rpn = self.rpn_shared(feature)
        scores = self.rpn_objectness(rpn).permute(0, 2, 3, 1).reshape(b, -1)
        deltas = self.rpn_delta(rpn).permute(0, 2, 3, 1).reshape(b, -1, 4)
        outputs = []
        for i in range(b):
            candidates = scores[i].sigmoid().topk(min(128, scores.size(1))).indices
            proposals = decode_boxes(deltas[i, candidates].detach(), anchors[candidates])
            keep = nms(proposals, scores[i, candidates].detach(), limit=topk)
            proposals = proposals[keep]
            pooled = self._roi_features(feature[i:i+1], proposals)
            hidden = self.roi_head(pooled)
            outputs.append({
                "anchors": anchors, "rpn_scores": scores[i], "rpn_deltas": deltas[i],
                "proposals": proposals, "roi_features": pooled, "feature": feature[i:i+1],
                "class_logits": self.class_head(hidden), "box_deltas": self.box_head(hidden),
            })
        return outputs

    def training_loss(self, image, gt_boxes, gt_labels):
        """gt_boxes: 每图 [M,4] 的归一化 xyxy；gt_labels: 1..classes。"""
        predictions = self(image)
        losses = []
        for pred, boxes, labels in zip(predictions, gt_boxes, gt_labels):
            if boxes.numel() == 0:
                rpn_cls = F.binary_cross_entropy_with_logits(
                    pred["rpn_scores"], torch.zeros_like(pred["rpn_scores"])
                )
                roi_cls = F.cross_entropy(
                    pred["class_logits"],
                    torch.zeros(pred["class_logits"].size(0), dtype=torch.long, device=image.device),
                )
                losses.append(rpn_cls + roi_cls + pred["rpn_deltas"].sum() * 0 + pred["box_deltas"].sum() * 0)
                continue
            overlap = box_iou(pred["anchors"], boxes)
            best_iou, best_gt = overlap.max(1)
            rpn_label = torch.full_like(best_iou, -1, dtype=torch.long)
            rpn_label[best_iou < .3] = 0
            rpn_label[best_iou >= .7] = 1
            forced_anchor, forced_gt = force_unique_matches(overlap)
            rpn_label[forced_anchor] = 1  # 每个 GT 至少有一个不同的正 anchor
            best_gt[forced_anchor] = forced_gt
            valid = rpn_label >= 0
            positive = rpn_label == 1
            rpn_cls = F.binary_cross_entropy_with_logits(pred["rpn_scores"][valid], rpn_label[valid].float())
            rpn_box = F.smooth_l1_loss(
                pred["rpn_deltas"][positive],
                encode_boxes(boxes[best_gt[positive]], pred["anchors"][positive]),
            )
            # 训练时加入 GT 框，保证分类/回归头在随机初始化时也收到正样本。
            roi_boxes = torch.cat([pred["proposals"], boxes.detach()], 0)
            hidden = self.roi_head(self._roi_features(pred["feature"], roi_boxes))
            class_logits, box_deltas = self.class_head(hidden), self.box_head(hidden)
            match_iou, match_gt = box_iou(roi_boxes, boxes).max(1)
            roi_label = torch.where(match_iou >= .5, labels[match_gt], torch.zeros_like(match_gt))
            roi_cls = F.cross_entropy(class_logits, roi_label)
            roi_positive = roi_label > 0
            roi_box = (F.smooth_l1_loss(
                box_deltas[roi_positive],
                encode_boxes(boxes[match_gt[roi_positive]], roi_boxes[roi_positive]),
            ) if roi_positive.any() else box_deltas.sum() * 0)
            losses.append(rpn_cls + rpn_box + roi_cls + roi_box)
        return torch.stack(losses).mean()


if __name__ == "__main__":
    model = FasterRCNN(classes=3, width=16)
    image = torch.randn(1, 3, 32, 32)
    result = model(image)[0]
    loss = model.training_loss(image, [torch.tensor([[.2, .2, .7, .7]])], [torch.tensor([1])])
    loss.backward()
    print("RPN anchors", tuple(result["anchors"].shape), "proposals", tuple(result["proposals"].shape),
          "RoI logits", tuple(result["class_logits"].shape), "loss", round(loss.item(), 4))
