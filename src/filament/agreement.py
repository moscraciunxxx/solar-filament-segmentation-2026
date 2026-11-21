"""Keep a mask only when a second model overlaps it.

The overlap cut is the contest true-positive rule, IoU > 0.5.
"""

from __future__ import annotations

from .disjoint_add import mask_iou
from .rle import decode_rle


def agreeing_rles(
    source_rles: list[str],
    witness_rles: list[str],
    min_iou: float = 0.5,
) -> list[str]:
    if not source_rles or not witness_rles:
        return []
    witnesses = [decode_rle(rle) for rle in witness_rles]
    kept: list[str] = []
    for rle in source_rles:
        mask = decode_rle(rle)
        if any(mask_iou(mask, other) > min_iou for other in witnesses):
            kept.append(rle)
    return kept
