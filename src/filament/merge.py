"""NMS-merge extra instances onto a frozen Slot-1 set. Keep Slot-1 intact."""

from __future__ import annotations

import numpy as np

from .rle import decode_rle

MERGE_IOU = 0.5


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float(inter / union) if union else 0.0


def extra_rles(
    slot1: list[str],
    candidates: list[str],
    iou_thr: float = MERGE_IOU,
) -> list[str]:
    """Candidates that merge_rles would add (IoU < thr vs every Slot-1 mask)."""
    if not candidates:
        return []
    slot_masks = [decode_rle(rle) for rle in slot1]
    extras: list[str] = []
    for rle in candidates:
        mask = decode_rle(rle)
        if slot_masks and max(mask_iou(mask, other) for other in slot_masks) >= iou_thr:
            continue
        extras.append(rle)
    return extras


def merge_rles(
    slot1: list[str],
    extra: list[str],
    iou_thr: float = MERGE_IOU,
) -> tuple[list[str], int]:
    """Keep every Slot-1 instance; add extra if IoU < thr vs all Slot-1 masks."""
    added_rles = extra_rles(slot1, extra, iou_thr=iou_thr)
    return list(slot1) + added_rles, len(added_rles)
