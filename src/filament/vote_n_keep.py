"""Keep a 2-of-3 vote component when it touches the 2048 mask or yolo11n.

The vote stays 2 of 3 among the 2048, year-shift, and all-annotator models.
A component that misses the 2048 mask is kept only when yolo11n overlaps it
above 0.5. This does not drop every such component.
"""

from __future__ import annotations

import numpy as np

from .rle import decode_rle
from .vote import components, vote_map

WITNESS_IOU = 0.5


def _best_iou(mask: np.ndarray, others: list[np.ndarray]) -> float:
    if not others:
        return 0.0
    flat = mask.ravel().astype(np.float32)
    stacked = np.stack([other.ravel() for other in others]).astype(np.float32)
    intersection = stacked @ flat
    union = stacked.sum(axis=1) + float(flat.sum()) - intersection
    overlaps = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return float(overlaps.max())


def _touches(mask: np.ndarray, parents: list[np.ndarray]) -> bool:
    if not parents:
        return False
    flat = mask.ravel().astype(np.float32)
    stacked = np.stack([parent.ravel() for parent in parents]).astype(np.float32)
    return bool(np.any(stacked @ flat > 0))


def vote_n_keep(
    parent_rles: list[str],
    shift_rles: list[str],
    all_rles: list[str],
    small_rles: list[str],
    witness_iou: float = WITNESS_IOU,
) -> list[str]:
    if not parent_rles and not shift_rles and not all_rles:
        return []
    voted = components(vote_map([parent_rles, shift_rles, all_rles]))
    parents = [decode_rle(rle) for rle in parent_rles]
    witnesses = [decode_rle(rle) for rle in small_rles]
    kept: list[str] = []
    for rle in voted:
        mask = decode_rle(rle)
        if _touches(mask, parents) or _best_iou(mask, witnesses) > witness_iou:
            kept.append(rle)
    return kept
