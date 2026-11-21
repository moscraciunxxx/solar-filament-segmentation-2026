"""Add a Mask R-CNN mask only when the year-shift model agrees.

Every 2048 mask is kept. A donor mask is appended only when it shares no
pixel with those masks and some witness mask overlaps it by more than 0.5.
The witness cut is fixed. This does not change a Mask R-CNN threshold.
"""

from __future__ import annotations

from .disjoint_add import mask_iou
from .paths import IMAGE_SIZE
from .rle import decode_rle

WITNESS_IOU = 0.5


def _best_iou(mask, others) -> float:
    if not others:
        return 0.0
    return max(mask_iou(mask, other) for other in others)


def rcnn_witness_add(
    parent_rles: list[str],
    donor_rles: list[str],
    witness_rles: list[str],
    witness_iou: float = WITNESS_IOU,
    image_size: int = IMAGE_SIZE,
) -> tuple[list[str], int]:
    """Return parent RLEs plus disjoint donor masks a witness confirms."""
    if not donor_rles or not witness_rles:
        return list(parent_rles), 0
    parent_masks = [decode_rle(rle, image_size, image_size) for rle in parent_rles]
    witness_masks = [decode_rle(rle, image_size, image_size) for rle in witness_rles]
    added: list[str] = []
    for rle in donor_rles:
        mask = decode_rle(rle, image_size, image_size)
        if any(mask_iou(mask, other) != 0.0 for other in parent_masks):
            continue
        if _best_iou(mask, witness_masks) > witness_iou:
            added.append(rle)
    return list(parent_rles) + added, len(added)
