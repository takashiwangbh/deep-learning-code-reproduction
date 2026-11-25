"""DETR：对象查询、编码器-解码器与精确的小规模二分匹配损失。"""
import itertools
import math
import torch
from torch import nn
from torch.nn import functional as F


def cxcywh_to_xyxy(box):
    return torch.cat([box[..., :2] - box[..., 2:] / 2, box[..., :2] + box[..., 2:] / 2], -1)


def pairwise_giou(a, b):
    a, b = cxcywh_to_xyxy(a), cxcywh_to_xyxy(b)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp_min(0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    union = area_a[:, None] + area_b[None] - inter
    outer_lt = torch.minimum(a[:, None, :2], b[None, :, :2])
    outer_rb = torch.maximum(a[:, None, 2:], b[None, :, 2:])
    enclosure = (outer_rb - outer_lt).clamp_min(0).prod(-1).clamp_min(1e-8)
    return inter / union.clamp_min(1e-8) - (enclosure - union) / enclosure


def optimal_assignment(cost):
    """小 Q/M 的穷举精确解；训练大模型时应换成 Hungarian 算法。"""
    q, m = cost.shape
    if m == 0:
        empty = torch.empty(0, dtype=torch.long, device=cost.device)
        return empty, empty
    if m > q or m > 4 or q > 10:
        raise ValueError("教学匹配器只支持目标<=4、查询<=10；大规模请用 Hungarian")
    detached = cost.detach()
    best_value, best_rows = float("inf"), None
    for rows in itertools.permutations(range(q), m):
        value = sum(float(detached[row, col]) for col, row in enumerate(rows))
        if value < best_value:
            best_value, best_rows = value, rows
    return torch.tensor(best_rows, device=cost.device), torch.arange(m, device=cost.device)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.width = heads, dim // heads
        self.q, self.k, self.v, self.out = [nn.Linear(dim, dim) for _ in range(4)]

    def forward(self, query, context):
        b, nq, d = query.shape
        nk = context.size(1)
        split = lambda z, n: z.reshape(b, n, self.heads, self.width).transpose(1, 2)
        q, k, v = split(self.q(query), nq), split(self.k(context), nk), split(self.v(context), nk)
        weights = ((q @ k.transpose(-2, -1)) / math.sqrt(self.width)).softmax(-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, nq, d))


class Encoder(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = Attention(dim, heads)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.ReLU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x))
        return self.norm2(x + self.ff(x))


class Decoder(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.self_attn, self.cross_attn = Attention(dim, heads), Attention(dim, heads)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.ReLU(), nn.Linear(4 * dim, dim))

    def forward(self, queries, memory):
        x = self.norm1(queries + self.self_attn(queries, queries))
        x = self.norm2(x + self.cross_attn(x, memory))
        return self.norm3(x + self.ff(x))


def position_2d(h, w, dim, device):
    if dim % 4:
        raise ValueError("dim 必须是 4 的整数倍")
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    freq = torch.exp(-math.log(10000) * torch.arange(dim // 4, device=device) / max(dim // 4 - 1, 1))
    px, py = x.flatten()[:, None] * freq, y.flatten()[:, None] * freq
    return torch.cat([px.sin(), px.cos(), py.sin(), py.cos()], -1)


class DETR(nn.Module):
    def __init__(self, classes=3, dim=64, heads=4, queries=8):
        super().__init__()
        self.classes = classes
        self.backbone = nn.Sequential(
            nn.Conv2d(3, dim // 2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(dim // 2, dim, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.encoder = Encoder(dim, heads)
        self.decoder = Decoder(dim, heads)
        self.queries = nn.Parameter(torch.randn(queries, dim) * .02)
        self.class_head = nn.Linear(dim, classes + 1)  # 最后一类为 no-object
        self.box_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 4))

    def forward(self, image):
        feature = self.backbone(image)
        b, d, h, w = feature.shape
        token = feature.flatten(2).transpose(1, 2) + position_2d(h, w, d, image.device)[None]
        memory = self.encoder(token)
        query = self.queries[None].expand(b, -1, -1)
        decoded = self.decoder(query, memory)
        return self.class_head(decoded), self.box_head(decoded).sigmoid()  # cx,cy,w,h

    def set_loss(self, image, gt_boxes, gt_labels):
        logits, boxes = self(image)
        losses = []
        for score, predicted, target_box, target_label in zip(logits, boxes, gt_boxes, gt_labels):
            if target_box.numel() == 0:
                no_object = torch.full((score.size(0),), self.classes, dtype=torch.long, device=score.device)
                losses.append(F.cross_entropy(score, no_object))
                continue
            cost = (-score.softmax(-1)[:, target_label]
                    + 5 * torch.cdist(predicted, target_box, p=1)
                    + 2 * (1 - pairwise_giou(predicted, target_box)))
            rows, cols = optimal_assignment(cost)
            class_target = torch.full((score.size(0),), self.classes, dtype=torch.long, device=score.device)
            class_target[rows] = target_label[cols]
            weight = torch.ones(self.classes + 1, device=score.device)
            weight[-1] = .1
            cls = F.cross_entropy(score, class_target, weight=weight)
            l1 = F.l1_loss(predicted[rows], target_box[cols])
            giou = 1 - pairwise_giou(predicted[rows], target_box[cols]).diag().mean()
            losses.append(cls + 5 * l1 + 2 * giou)
        return torch.stack(losses).mean()


if __name__ == "__main__":
    model = DETR(classes=3, dim=32, queries=6)
    image = torch.randn(1, 3, 32, 32)
    scores, boxes = model(image)
    loss = model.set_loss(image, [torch.tensor([[.5, .5, .3, .3], [.2, .2, .1, .1]])], [torch.tensor([0, 2])])
    loss.backward()
    print("DETR classes", tuple(scores.shape), "boxes", tuple(boxes.shape), "loss", round(loss.item(), 4))
