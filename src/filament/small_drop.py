"""Drop a mask whose area is below a fixed pixel count.

The count is 96, the minimum already used when splitting a semantic blob.
Nothing is added and no mask is grown.
"""

from __future__ import annotations

import numpy as np

from .rle import decode_rle

MIN_AREA = 96


def drop_small_rles(rles: list[str], min_area: int = MIN_AREA) -> list[str]:
    kept: list[str] = []
    for rle in rles:
        area = int(np.count_nonzero(decode_rle(rle)))
        if area >= min_area:
            kept.append(rle)
    return kept
