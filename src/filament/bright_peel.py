"""Remove a bright outer layer from a mask.

The interior median ignores that outer layer. A boundary pixel more than
12 gray levels lighter than the interior is halo and is dropped. One pass.
This does not add pixels and does not scale the mask.
"""

from __future__ import annotations

import cv2
import numpy as np

from .rle import decode_rle, encode_rle

MARGIN = 12
MIN_KEEP = 32


def bright_peel(image: np.ndarray, mask: np.ndarray, margin: int = MARGIN) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    interior = cv2.erode(binary, kernel)
    if int(interior.sum()) < MIN_KEEP:
        return binary
    level = float(np.median(gray[interior > 0])) + margin
    boundary = (binary > 0) & (interior == 0)
    peeled = binary.copy()
    peeled[boundary & (gray > level)] = 0
    if int(peeled.sum()) < MIN_KEEP:
        return binary
    return peeled


def bright_peel_rles(image: np.ndarray, rles: list[str]) -> list[str]:
    return [encode_rle(bright_peel(image, decode_rle(rle))) for rle in rles]
