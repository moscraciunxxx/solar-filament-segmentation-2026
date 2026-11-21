"""Add a source mask only when three fixed conditions all hold.

It misses every parent mask, its area is at least 96 pixels, and a witness
mask overlaps it at IoU > 0.5. None of those cuts is swept.
"""

from __future__ import annotations

import numpy as np

from .agreement import agreeing_rles
from .disjoint_add import disjoint_add
from .rle import decode_rle
from .small_drop import MIN_AREA

WITNESS_IOU = 0.5


def strict_add(
    parent_rles: list[str],
    source_rles: list[str],
    witness_rles: list[str],
    min_area: int = MIN_AREA,
    min_iou: float = WITNESS_IOU,
) -> tuple[list[str], int]:
    large = [
        rle
        for rle in source_rles
        if int(np.count_nonzero(decode_rle(rle))) >= min_area
    ]
    agreed = agreeing_rles(large, witness_rles, min_iou)
    return disjoint_add(parent_rles, agreed)
