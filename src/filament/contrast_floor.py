"""Drop a mask fainter than the training-filament contrast floor.

The floor is the 10th percentile of medoid-filament contrast on folds other
than 0. A mask we cannot measure is kept. This does not use the 12-gray
add-gate and it does not edit the boundary of a mask that stays.
"""

from __future__ import annotations

import cv2
import numpy as np

from .rle import decode_rle

RING = 31
MIN_INSIDE = 32
MIN_RING = 32


def mask_contrast(mask: np.ndarray, image: np.ndarray, ring: int = RING) -> float | None:
    """Mean ring gray minus mean interior gray. None when the ring is too small."""
    inside = image[mask > 0]
    if inside.size < MIN_INSIDE:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring, ring))
    dilated = cv2.dilate((mask > 0).astype(np.uint8), kernel)
    ring_pixels = image[(dilated > 0) & (mask == 0)]
    if ring_pixels.size < MIN_RING:
        return None
    return float(ring_pixels.mean() - inside.mean())


def drop_faint_rles(
    rles: list[str],
    image: np.ndarray,
    floor: float,
    image_size: int | None = None,
) -> tuple[list[str], int]:
    """Keep every mask at least as dark as ``floor``."""
    height, width = image.shape[:2]
    kept: list[str] = []
    dropped = 0
    for rle in rles:
        mask = decode_rle(rle, image_size or height, image_size or width)
        contrast = mask_contrast(mask, image)
        if contrast is not None and contrast < floor:
            dropped += 1
            continue
        kept.append(rle)
    return kept, dropped
