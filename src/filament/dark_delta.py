"""Keep a mask only when it is darker than the ring around it.

The margin is fixed at 12 gray levels. It is not fit on a score.
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_DELTA = 12
MIN_INSIDE = 64
RING = 21


def dark_enough(
    mask: np.ndarray,
    image: np.ndarray,
    min_delta: float = MIN_DELTA,
) -> bool:
    inside = image[mask > 0]
    if inside.size < MIN_INSIDE:
        return False
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RING, RING))
    dilated = cv2.dilate((mask > 0).astype(np.uint8), kernel)
    ring = image[(dilated > 0) & (mask == 0)]
    if ring.size < 32:
        return False
    return float(ring.mean() - inside.mean()) >= min_delta
