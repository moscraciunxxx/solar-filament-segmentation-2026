"""Close small gaps inside one mask. The kernel is a 7-pixel ellipse, once.

This does not grow tips along an axis, fill only enclosed holes, or add a
second model's instances.
"""

from __future__ import annotations

import cv2
import numpy as np

from .rle import decode_rle, encode_rle

KERNEL = 7


def close_mask(mask: np.ndarray, kernel: int = KERNEL) -> np.ndarray:
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    binary = (mask > 0).astype(np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)


def close_rles(rles: list[str], kernel: int = KERNEL) -> list[str]:
    return [encode_rle(close_mask(decode_rle(rle), kernel)) for rle in rles]
