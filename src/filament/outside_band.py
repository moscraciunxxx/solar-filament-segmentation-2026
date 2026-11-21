"""Edit masks outside the 35-pair partner band.

A parent mask is dropped only when the year-shift model, yolo11n, and the
all-annotator model all share zero pixels with it. An all-annotator mask is
added only when it shares zero pixels with the parent and a year-shift or
yolo11n mask overlaps it above 0.5. Partial parent/shift pairs are left
unchanged. This is not a pixel vote and not a disjoint dump.
"""

from __future__ import annotations

import numpy as np

from .contrast_floor import mask_contrast
from .rle import decode_rle, encode_rle

WITNESS_IOU = 0.5


def _best_iou(mask: np.ndarray, others: list[np.ndarray]) -> float:
    if not others:
        return 0.0
    flat = mask.ravel().astype(np.float32)
    stacked = np.stack([other.ravel() for other in others]).astype(np.float32)
    intersection = stacked @ flat
    union = stacked.sum(axis=1) + flat.sum() - intersection
    overlaps = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return float(overlaps.max())


def _unseen(
    mask: np.ndarray,
    shift_masks: list[np.ndarray],
    small_masks: list[np.ndarray],
    all_masks: list[np.ndarray],
) -> bool:
    return (
        _best_iou(mask, shift_masks) == 0.0
        and _best_iou(mask, small_masks) == 0.0
        and _best_iou(mask, all_masks) == 0.0
    )


def drop_unseen_masks(
    parent_masks: list[np.ndarray],
    shift_masks: list[np.ndarray],
    small_masks: list[np.ndarray],
    all_masks: list[np.ndarray],
) -> list[np.ndarray]:
    """Drop a mask only when shift, yolo11n, and all-annotator all miss it.

    No mask is added. The zero-pixel test is the locked one.
    """
    return [
        mask.copy()
        for mask in parent_masks
        if not _unseen(mask, shift_masks, small_masks, all_masks)
    ]


def drop_faint_unseen_masks(
    parent_masks: list[np.ndarray],
    shift_masks: list[np.ndarray],
    small_masks: list[np.ndarray],
    all_masks: list[np.ndarray],
    image: np.ndarray,
    floor: float,
) -> list[np.ndarray]:
    """Drop an unseen mask only when its contrast is below the locked floor.

    A mask another model touches stays. A mask whose ring cannot be measured
    stays. The floor and the 31-pixel ring are not changed here.
    """
    kept: list[np.ndarray] = []
    for mask in parent_masks:
        if not _unseen(mask, shift_masks, small_masks, all_masks):
            kept.append(mask.copy())
            continue
        contrast = mask_contrast(mask, image)
        if contrast is not None and contrast < floor:
            continue
        kept.append(mask.copy())
    return kept


def drop_unseen_unless_touched(
    parent_masks: list[np.ndarray],
    shift_masks: list[np.ndarray],
    small_masks: list[np.ndarray],
    all_masks: list[np.ndarray],
    large_masks: list[np.ndarray],
) -> list[np.ndarray]:
    """Drop an unseen mask only when the large model also misses it.

    A mask the year-shift model, yolo11n, or the all-annotator model touches
    stays. An unseen mask stays when the large model shares a pixel. Nothing
    is added. The zero-pixel test is unchanged.
    """
    kept: list[np.ndarray] = []
    for mask in parent_masks:
        if not _unseen(mask, shift_masks, small_masks, all_masks):
            kept.append(mask.copy())
            continue
        if _best_iou(mask, large_masks) > 0.0:
            kept.append(mask.copy())
    return kept


def drop_unseen_unless_both_touch(
    parent_masks: list[np.ndarray],
    shift_masks: list[np.ndarray],
    small_masks: list[np.ndarray],
    all_masks: list[np.ndarray],
    large_masks: list[np.ndarray],
    rcnn_masks: list[np.ndarray],
) -> list[np.ndarray]:
    """Drop an unseen mask unless two extra models both share a pixel with it.

    One extra model is not enough. Nothing is added. The zero-pixel test on
    the year-shift model, yolo11n, and the all-annotator model is unchanged.
    """
    kept: list[np.ndarray] = []
    for mask in parent_masks:
        if not _unseen(mask, shift_masks, small_masks, all_masks):
            kept.append(mask.copy())
            continue
        if _best_iou(mask, large_masks) > 0.0 and _best_iou(mask, rcnn_masks) > 0.0:
            kept.append(mask.copy())
    return kept


def drop_unseen_unless_triple_touch(
    parent_masks: list[np.ndarray],
    shift_masks: list[np.ndarray],
    small_masks: list[np.ndarray],
    all_masks: list[np.ndarray],
    large_masks: list[np.ndarray],
    rcnn_masks: list[np.ndarray],
    full_masks: list[np.ndarray],
) -> list[np.ndarray]:
    """Drop an unseen mask unless three extra models all share a pixel with it.

    Two extra models are not enough. Nothing is added. The zero-pixel test
    on the year-shift model, yolo11n, and the all-annotator model is unchanged.
    """
    kept: list[np.ndarray] = []
    for mask in parent_masks:
        if not _unseen(mask, shift_masks, small_masks, all_masks):
            kept.append(mask.copy())
            continue
        if (
            _best_iou(mask, large_masks) > 0.0
            and _best_iou(mask, rcnn_masks) > 0.0
            and _best_iou(mask, full_masks) > 0.0
        ):
            kept.append(mask.copy())
    return kept


def edit_outside_band(
    parent_masks: list[np.ndarray],
    shift_masks: list[np.ndarray],
    small_masks: list[np.ndarray],
    all_masks: list[np.ndarray],
    witness_iou: float = WITNESS_IOU,
) -> list[np.ndarray]:
    kept: list[np.ndarray] = []
    for mask in parent_masks:
        if not _unseen(mask, shift_masks, small_masks, all_masks):
            kept.append(mask.copy())
    for mask in all_masks:
        if _best_iou(mask, parent_masks) != 0.0:
            continue
        if (
            _best_iou(mask, shift_masks) > witness_iou
            or _best_iou(mask, small_masks) > witness_iou
        ):
            kept.append(mask.copy())
    return kept


def edit_outside_rles(
    parent_rles: list[str],
    shift_rles: list[str],
    small_rles: list[str],
    all_rles: list[str],
) -> tuple[list[str], int, int]:
    parent_masks = [decode_rle(rle) for rle in parent_rles]
    shift_masks = [decode_rle(rle) for rle in shift_rles]
    small_masks = [decode_rle(rle) for rle in small_rles]
    all_masks = [decode_rle(rle) for rle in all_rles]
    edited = edit_outside_band(parent_masks, shift_masks, small_masks, all_masks)
    n_dropped = sum(
        1 for mask in parent_masks if _unseen(mask, shift_masks, small_masks, all_masks)
    )
    n_added = len(edited) - (len(parent_masks) - n_dropped)
    return [encode_rle(mask) for mask in edited], n_dropped, n_added
