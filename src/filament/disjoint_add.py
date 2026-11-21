"""Keep a frozen parent set and add only pixel-disjoint candidates.

This is not a full OR. Every parent mask is kept. A candidate is appended
only when its IoU with every parent mask is exactly 0. The cut is fixed:
there is no overlap threshold and no sweep. Candidates are not compared
to each other.
"""

from __future__ import annotations

import numpy as np

from .paths import IMAGE_SIZE
from .rle import decode_rle


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    inter = int(np.count_nonzero(aa & bb))
    union = int(np.count_nonzero(aa | bb))
    if union == 0:
        return 0.0
    return inter / union


def disjoint_add(
    parent_rles: list[str],
    candidate_rles: list[str],
    image_size: int = IMAGE_SIZE,
) -> tuple[list[str], int]:
    """Return parent RLEs plus candidates whose IoU is 0 against all of them."""
    if not candidate_rles:
        return list(parent_rles), 0
    parent_masks = [decode_rle(rle, image_size, image_size) for rle in parent_rles]
    added: list[str] = []
    for rle in candidate_rles:
        mask = decode_rle(rle, image_size, image_size)
        if any(mask_iou(mask, other) != 0.0 for other in parent_masks):
            continue
        added.append(rle)
    return list(parent_rles) + added, len(added)
