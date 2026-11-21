"""Grow a parent mask into a partner mask only when the two partly agree.

The partner is the year-shift model. Agreement in [0.10, 0.50) means the
two outlines are the same object drawn differently. Pixels are taken from
the partner only inside a 24-pixel clip of the parent, about one train
filament width. A disjoint partner is not added. A near-copy is left alone.
This is not a pixel vote and not a uniform scale.
"""

from __future__ import annotations

import cv2
import numpy as np

from .rle import decode_rle, encode_rle

LO = 0.10
HI = 0.50
CLIP = 24


def _iou_matrix(left: list[np.ndarray], right: list[np.ndarray]) -> np.ndarray:
    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=np.float32)
    stacked_left = np.stack([mask.ravel() for mask in left]).astype(np.float32)
    stacked_right = np.stack([mask.ravel() for mask in right]).astype(np.float32)
    intersection = stacked_left @ stacked_right.T
    union = stacked_left.sum(axis=1)[:, None] + stacked_right.sum(axis=1)[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def grow_masks(
    parent_masks: list[np.ndarray],
    partner_masks: list[np.ndarray],
    lo: float = LO,
    hi: float = HI,
    clip: int = CLIP,
) -> list[np.ndarray]:
    ious = _iou_matrix(parent_masks, partner_masks)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clip * 2 + 1, clip * 2 + 1))
    grown: list[np.ndarray] = []
    for index, mask in enumerate(parent_masks):
        if partner_masks:
            partner_index = int(np.argmax(ious[index]))
            overlap = float(ious[index, partner_index])
        else:
            partner_index = -1
            overlap = 0.0
        if partner_index >= 0 and lo <= overlap < hi:
            zone = cv2.dilate(mask, kernel)
            extra = (zone > 0) & (partner_masks[partner_index] > 0) & (mask == 0)
            merged = mask.copy()
            merged[extra] = 1
            grown.append(merged)
        else:
            grown.append(mask.copy())
    return grown


def replace_partial_masks(
    parent_masks: list[np.ndarray],
    partner_masks: list[np.ndarray],
    lo: float = LO,
    hi: float = HI,
) -> list[np.ndarray]:
    """Swap in the whole partner when overlap is partial. No clip, no new instance."""
    ious = _iou_matrix(parent_masks, partner_masks)
    replaced: list[np.ndarray] = []
    for index, mask in enumerate(parent_masks):
        if partner_masks:
            partner_index = int(np.argmax(ious[index]))
            overlap = float(ious[index, partner_index])
        else:
            partner_index = -1
            overlap = 0.0
        if partner_index >= 0 and lo <= overlap < hi:
            replaced.append(partner_masks[partner_index].copy())
        else:
            replaced.append(mask.copy())
    return replaced


def union_partial_masks(
    parent_masks: list[np.ndarray],
    partner_masks: list[np.ndarray],
    lo: float = LO,
    hi: float = HI,
) -> list[np.ndarray]:
    """Keep every pixel of both masks when their overlap is partial."""
    ious = _iou_matrix(parent_masks, partner_masks)
    united: list[np.ndarray] = []
    for index, mask in enumerate(parent_masks):
        if partner_masks:
            partner_index = int(np.argmax(ious[index]))
            overlap = float(ious[index, partner_index])
        else:
            partner_index = -1
            overlap = 0.0
        if partner_index >= 0 and lo <= overlap < hi:
            united.append(np.maximum(mask, partner_masks[partner_index]))
        else:
            united.append(mask.copy())
    return united


def grow_rles(parent_rles: list[str], partner_rles: list[str]) -> list[str]:
    parent_masks = [decode_rle(rle) for rle in parent_rles]
    partner_masks = [decode_rle(rle) for rle in partner_rles]
    return [encode_rle(mask) for mask in grow_masks(parent_masks, partner_masks)]


def replace_partial_rles(parent_rles: list[str], partner_rles: list[str]) -> list[str]:
    parent_masks = [decode_rle(rle) for rle in parent_rles]
    partner_masks = [decode_rle(rle) for rle in partner_rles]
    return [encode_rle(mask) for mask in replace_partial_masks(parent_masks, partner_masks)]


def union_partial_rles(parent_rles: list[str], partner_rles: list[str]) -> list[str]:
    parent_masks = [decode_rle(rle) for rle in parent_rles]
    partner_masks = [decode_rle(rle) for rle in partner_rles]
    return [encode_rle(mask) for mask in union_partial_masks(parent_masks, partner_masks)]
