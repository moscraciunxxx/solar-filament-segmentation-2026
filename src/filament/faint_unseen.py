"""Drop a faint 2048 mask only when yolo11n does not see it.

The contrast floor stays the training 10th percentile. The witness cut
stays 0.5. A faint mask that yolo11n overlaps is kept. A mask we cannot
measure is kept.
"""

from __future__ import annotations

import numpy as np

from .contrast_floor import mask_contrast
from .disjoint_add import mask_iou
from .rle import decode_rle

WITNESS_IOU = 0.5


def _best_iou(mask: np.ndarray, others: list[np.ndarray]) -> float:
    if not others:
        return 0.0
    return max(mask_iou(mask, other) for other in others)


def drop_faint_unseen(
    parent_rles: list[str],
    image: np.ndarray,
    witness_rles: list[str],
    floor: float,
    witness_iou: float = WITNESS_IOU,
) -> tuple[list[str], int]:
    """Return parent masks after the faint-and-unseen drop."""
    height, width = image.shape[:2]
    witnesses = [decode_rle(rle, height, width) for rle in witness_rles]
    kept: list[str] = []
    dropped = 0
    for rle in parent_rles:
        mask = decode_rle(rle, height, width)
        contrast = mask_contrast(mask, image)
        if (
            contrast is not None
            and contrast < floor
            and _best_iou(mask, witnesses) <= witness_iou
        ):
            dropped += 1
            continue
        kept.append(rle)
    return kept, dropped
