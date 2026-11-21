"""CondInst dynamic-filter instance head.

Tian, Shen, Chen, He: each positive location generates the weights of a
small mask conv and applies it to a shared mask feature plus relative
coordinates. This is not RoIAlign and not a YOLO prototype head.

Training and scoring stay on contest train JPEGs. This module does not
read the test image directory.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MASK_DIM = 8
WIDTH = 8
IN_CH = MASK_DIM + 2
STRIDE = 8
PUBLIC_GAP = 0.069
PUBLIC_BAR = 0.6
PARENT_PQ = 0.4387235694048039
PARENT_TP = 949
MAX_MASK_POS = 48


def controller_dim(in_ch: int = IN_CH, width: int = WIDTH) -> int:
    """Parameter count of three 1x1 convs: in→width, width→width, width→1."""
    return in_ch * width + width + width * width + width + width + 1


CONTROLLER_DIM = controller_dim()


def should_promote(pq: float, tp: int, *, same_hit_set: bool) -> bool:
    """Upload gate. Forecast uses the measured 2048-parent public gap."""
    if (pq - PUBLIC_GAP) > PUBLIC_BAR:
        return True
    if pq < 0.47:
        return False
    if tp != PARENT_TP:
        return True
    return not same_hit_set


def apply_dynamic_mask(
    mask_feat: torch.Tensor,
    center_yx: tuple[int, int],
    params: torch.Tensor,
) -> torch.Tensor:
    """Run one generated 3-layer 1x1 mask head. Returns (H, W) logits."""
    channels, height, width = mask_feat.shape
    if channels != MASK_DIM:
        raise ValueError(f"mask feature has {channels} channels, expected {MASK_DIM}")
    if params.numel() != CONTROLLER_DIM:
        raise ValueError(f"controller has {params.numel()} values, expected {CONTROLLER_DIM}")
    ys = torch.arange(height, device=mask_feat.device, dtype=mask_feat.dtype)
    xs = torch.arange(width, device=mask_feat.device, dtype=mask_feat.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    cy, cx = center_yx
    denom_y = float(max(height, 1))
    denom_x = float(max(width, 1))
    rel = torch.stack(((grid_y - cy) / denom_y, (grid_x - cx) / denom_x), dim=0)
    hidden = torch.cat((mask_feat, rel), dim=0).unsqueeze(0)
    cursor = 0

    def take(count: int) -> torch.Tensor:
        nonlocal cursor
        piece = params[cursor : cursor + count]
        cursor += count
        return piece

    weight_1 = take(IN_CH * WIDTH).view(WIDTH, IN_CH, 1, 1)
    bias_1 = take(WIDTH)
    weight_2 = take(WIDTH * WIDTH).view(WIDTH, WIDTH, 1, 1)
    bias_2 = take(WIDTH)
    weight_3 = take(WIDTH).view(1, WIDTH, 1, 1)
    bias_3 = take(1)
    if cursor != CONTROLLER_DIM:
        raise RuntimeError("controller slice did not consume every parameter")
    layer_1 = F.relu(F.conv2d(hidden, weight_1, bias_1))
    layer_2 = F.relu(F.conv2d(layer_1, weight_2, bias_2))
    return F.conv2d(layer_2, weight_3, bias_3)[0, 0]


def _focal(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = logit.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    p_t = prob * target + (1 - prob) * (1 - target)
    alpha_t = 0.25 * target + 0.75 * (1 - target)
    loss = alpha_t * (1 - p_t).pow(2) * ce
    positives = target.sum()
    if float(positives) <= 0:
        return loss.mean()
    return loss.sum() / positives.clamp(min=1)


def _giou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.sum()
    left = torch.maximum(pred[:, 0], target[:, 0])
    top = torch.maximum(pred[:, 1], target[:, 1])
    right = torch.minimum(pred[:, 2], target[:, 2])
    bottom = torch.minimum(pred[:, 3], target[:, 3])
    inter = (right - left).clamp(min=0) * (bottom - top).clamp(min=0)
    area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_t = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    union = (area_p + area_t - inter).clamp(min=1e-6)
    iou = inter / union
    enc_l = torch.minimum(pred[:, 0], target[:, 0])
    enc_t = torch.minimum(pred[:, 1], target[:, 1])
    enc_r = torch.maximum(pred[:, 2], target[:, 2])
    enc_b = torch.maximum(pred[:, 3], target[:, 3])
    area_c = (enc_r - enc_l).clamp(min=0) * (enc_b - enc_t).clamp(min=0)
    giou = iou - (area_c - union) / area_c.clamp(min=1e-6)
    return (1 - giou).mean()


def _dice(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = logit.sigmoid()
    inter = (prob * target).sum()
    denom = prob.sum() + target.sum()
    return 1 - (2 * inter + 1) / (denom + 1)


def assign_owner(instances: list[dict], grid: int) -> torch.Tensor:
    """Feature-cell owner. Smallest mask wins a tie. -1 is background."""
    device = instances[0]["mask"].device
    owner = torch.full((grid, grid), -1, dtype=torch.long, device=device)
    area = torch.full((grid, grid), 1e18, device=device)
    for index, inst in enumerate(instances):
        mask = inst["mask"]
        small = F.interpolate(mask.float()[None, None], size=(grid, grid), mode="nearest")[0, 0] > 0.5
        inst_area = float(mask.sum())
        better = small & (inst_area < area)
        owner = torch.where(better, torch.full_like(owner, index), owner)
        area = torch.where(better, torch.full_like(area, inst_area), area)
    return owner


def sample_positives(
    instances: list[dict],
    grid: int,
    stride: int = STRIDE,
    limit: int = MAX_MASK_POS,
) -> tuple[list[tuple[int, int, int]], torch.Tensor]:
    """Mask-head samples spread along each filament, plus the owner map.

    Classification uses every interior cell. The mask head uses a uniform
    subset, because supervising only the cells nearest the centroid taught
    the class head that the rest of the filament was background.
    """
    if grid < 1 or not instances:
        device = instances[0]["mask"].device if instances else torch.device("cpu")
        return [], torch.full((max(grid, 1), max(grid, 1)), -1, dtype=torch.long, device=device)
    owner = assign_owner(instances, grid)
    chosen: list[tuple[int, int, int]] = []
    quota = max(1, limit // max(len(instances), 1))
    for index in range(len(instances)):
        ys, xs = torch.where(owner == index)
        count = int(ys.numel())
        if count == 0:
            continue
        step = max(1, count // quota)
        for pick in range(0, count, step):
            chosen.append((index, int(ys[pick]), int(xs[pick])))
            if len(chosen) >= limit:
                return chosen, owner
    return chosen, owner


def condinst_loss(outputs: dict, instances: list[dict], stride: int = STRIDE) -> dict[str, torch.Tensor]:
    """One-image loss. outputs are unbatched maps at stride."""
    logit = outputs["logit"]
    centerness = outputs["centerness"]
    ltrb = outputs["ltrb"]
    controller = outputs["controller"]
    mask_feat = outputs["mask_feat"]
    if logit.dim() != 3 or ltrb.dim() != 3 or mask_feat.dim() != 3:
        raise ValueError("condinst_loss expects unbatched maps with a channel axis")
    grid = int(logit.shape[-1])
    target_cls = torch.zeros_like(logit)
    points, owner = sample_positives(instances, grid, stride=stride)
    body = owner >= 0
    target_cls[0][body] = 1
    box_pred = []
    box_tgt = []
    mask_losses = []
    for inst_index, fy, fx in points:
        box = instances[inst_index]["box"]
        x1 = box[0] / stride
        y1 = box[1] / stride
        x2 = box[2] / stride
        y2 = box[3] / stride
        pred_box = torch.stack(
            (
                fx - ltrb[0, fy, fx],
                fy - ltrb[1, fy, fx],
                fx + ltrb[2, fy, fx],
                fy + ltrb[3, fy, fx],
            )
        )
        true_box = torch.stack((x1, y1, x2, y2))
        box_pred.append(pred_box)
        box_tgt.append(true_box)
        pred_mask = apply_dynamic_mask(mask_feat, (fy, fx), controller[:, fy, fx])
        gt_small = F.interpolate(
            instances[inst_index]["mask"].float()[None, None],
            size=pred_mask.shape,
            mode="nearest",
        )[0, 0]
        mask_losses.append(_dice(pred_mask, gt_small) + F.binary_cross_entropy_with_logits(pred_mask, gt_small))
    cls_loss = _focal(logit, target_cls)
    # Box-centerness is high at the box center. Curved filaments miss that
    # point, and the decode multiplies by centerness, so the body never
    # cleared 0.35. The body itself is the positive.
    ctr_loss = F.binary_cross_entropy_with_logits(centerness[0], body.float())
    if box_pred:
        box_loss = _giou(torch.stack(box_pred), torch.stack(box_tgt))
        mask_loss = torch.stack(mask_losses).mean()
    else:
        zero = logit.sum() * 0
        box_loss = zero
        mask_loss = zero
    total = cls_loss + box_loss + ctr_loss + mask_loss
    return {
        "total": total,
        "cls": cls_loss.detach(),
        "box": box_loss.detach(),
        "ctr": ctr_loss.detach(),
        "mask": mask_loss.detach(),
        "n_pos": logit.new_tensor(float(body.sum())),
    }


class CondInst(nn.Module):
    """ResNet-18 stride-8 stem, or a tiny stem for tests. Dynamic mask head."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        if backbone not in {"resnet18", "tiny"}:
            raise ValueError("backbone must be resnet18 or tiny")
        self.backbone_name = backbone
        if backbone == "tiny":
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 128, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet18(weights=weights)
            self.stem = nn.Sequential(
                net.conv1,
                net.bn1,
                net.relu,
                net.maxpool,
                net.layer1,
                net.layer2,
            )
        self.mask_tower = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, MASK_DIM, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(128, 1, 3, padding=1)
        self.ctr_head = nn.Conv2d(128, 1, 3, padding=1)
        self.box_head = nn.Conv2d(128, 4, 3, padding=1)
        self.ctrl_head = nn.Conv2d(128, CONTROLLER_DIM, 3, padding=1)
        nn.init.constant_(self.cls_head.bias, -4.6)
        nn.init.normal_(self.ctrl_head.weight, std=0.001)
        nn.init.zeros_(self.ctrl_head.bias)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.stem(images)
        return {
            "logit": self.cls_head(feat),
            "centerness": self.ctr_head(feat),
            "ltrb": self.box_head(feat).exp(),
            "controller": self.ctrl_head(feat),
            "mask_feat": self.mask_tower(feat),
        }


def predict_maps(
    model: CondInst,
    image: torch.Tensor,
    *,
    conf: float,
    imgsz: int,
    out_size: int,
    min_area: int = 96,
    max_det: int = 80,
) -> list[torch.Tensor]:
    """Decode one CHW image into a list of out_size binary masks on CPU."""
    model.eval()
    with torch.no_grad():
        outputs = model(image.unsqueeze(0))
    logit = outputs["logit"][0, 0]
    score = logit.sigmoid() * outputs["centerness"][0, 0].sigmoid()
    keep = torch.nonzero(score >= conf, as_tuple=False)
    if keep.numel() == 0:
        return []
    order = torch.argsort(score[keep[:, 0], keep[:, 1]], descending=True)[:max_det]
    keep = keep[order]
    stride = STRIDE
    scale = out_size / float(imgsz)
    boxes = []
    scores = []
    masks = []
    feat = outputs["mask_feat"][0]
    controller = outputs["controller"][0]
    ltrb = outputs["ltrb"][0]
    for fy, fx in keep.tolist():
        left, top, right, bottom = ltrb[:, fy, fx]
        x1 = (fx - float(left)) * stride * scale
        y1 = (fy - float(top)) * stride * scale
        x2 = (fx + float(right)) * stride * scale
        y2 = (fy + float(bottom)) * stride * scale
        if x2 <= x1 or y2 <= y1:
            continue
        logit_mask = apply_dynamic_mask(feat, (fy, fx), controller[:, fy, fx])
        up = F.interpolate(logit_mask[None, None], size=(out_size, out_size), mode="bilinear", align_corners=False)[0, 0]
        binary = up.sigmoid() >= 0.5
        if int(binary.sum()) < min_area:
            continue
        boxes.append([x1, y1, x2, y2])
        scores.append(float(score[fy, fx]))
        masks.append(binary)
    if not boxes:
        return []
    box_t = torch.tensor(boxes, dtype=torch.float32)
    score_t = torch.tensor(scores, dtype=torch.float32)
    from torchvision.ops import nms

    kept = nms(box_t, score_t, 0.5)
    return [masks[int(i)].to(dtype=torch.uint8).cpu() for i in kept.tolist()]
