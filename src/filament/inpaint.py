"""Paint parent masks with the median of the pixels they do not cover."""

from __future__ import annotations

import numpy as np

from .rle import decode_rle


def inpaint_parent(image: np.ndarray, rles: list[str]) -> np.ndarray:
    out = image.copy()
    occupied = np.zeros(image.shape[:2], dtype=np.uint8)
    for rle in rles:
        occupied = np.maximum(occupied, decode_rle(rle))
    outside = occupied == 0
    fill = int(np.median(image[outside])) if int(outside.sum()) else int(np.median(image))
    out[occupied > 0] = fill
    return out
