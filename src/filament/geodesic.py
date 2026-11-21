"""Grow a mask only through an already-dark map, for a fixed number of steps.

No new component is created. The step count is 24 and the dark cut is the
85th percentile of positive residual. Neither is swept.
"""

from __future__ import annotations

import cv2
import numpy as np

from .classical import dark_residual_channel
from .rle import decode_rle, encode_rle

STEPS = 24
DARK_PERCENTILE = 85.0


def geodesic_extend(mask: np.ndarray, dark: np.ndarray, steps: int = STEPS) -> np.ndarray:
    current = (mask > 0).astype(np.uint8)
    allowed = (dark > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(steps):
        grown = cv2.dilate(current, kernel)
        nxt = np.where((grown > 0) & ((current > 0) | (allowed > 0)), 1, 0).astype(np.uint8)
        if np.array_equal(nxt, current):
            break
        current = nxt
    return current


def dark_map(image: np.ndarray, percentile: float = DARK_PERCENTILE) -> np.ndarray:
    residual = dark_residual_channel(image)
    positive = residual[residual > 0]
    if positive.size == 0:
        return np.zeros_like(residual)
    threshold = float(np.percentile(positive, percentile))
    return (residual >= threshold).astype(np.uint8)


def extend_rles(image: np.ndarray, rles: list[str], steps: int = STEPS) -> list[str]:
    dark = dark_map(image)
    return [encode_rle(geodesic_extend(decode_rle(rle), dark, steps)) for rle in rles]
