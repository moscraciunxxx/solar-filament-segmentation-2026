"""Memory-only patch for ultralytics 8.4.146 v8SegmentationLoss.calculate_segmentation_loss (overlap=True path).
Builds the per-image gt_mask INSIDE a checkpointed function, so neither the (n_fg,H,W) float gt_mask nor the logits/BCE
are kept for backward. Identical math to the original."""
import torch, torch.utils.checkpoint as ck
from ultralytics.utils import loss as L
from ultralytics.utils.ops import xyxy2xywh
_sml = L.v8SegmentationLoss.single_mask_loss
def _img_loss(mask_i, mask_idx, pred, proto, xyxy, area):
    gt_mask = (mask_i == (mask_idx + 1).view(-1, 1, 1)).float()
    return _sml(gt_mask, pred, proto, xyxy, area)
def calculate_segmentation_loss(self, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz):
    assert self.overlap, "patch covers overlap=True only"
    _, _, mask_h, mask_w = proto.shape
    loss = 0
    tbn = target_bboxes / imgsz[[1, 0, 1, 0]]
    marea = xyxy2xywh(tbn)[..., 2:].prod(2)
    mxyxy = tbn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)
    for i, (fg, tgi, pm, pr, xy, ar) in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea)):
        if fg.any():
            loss += ck.checkpoint(_img_loss, masks[i], tgi[fg], pm[fg], pr, xy[fg], ar[fg], use_reentrant=False)
        else:
            loss += (proto * 0).sum() + (pred_masks * 0).sum()
    return loss / fg_mask.sum()
