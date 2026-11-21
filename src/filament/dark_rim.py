"""Add a 3-pixel rim where the new pixels are about as dark as the mask.

The margin is 8 gray levels, the faint end of train-filament contrast.
Bright neighbors stay out. Nothing farther than 3 pixels is added.
This is not a morphological close and not a new-instance proposer.
"""

from __future__ import annotations

import cv2
import numpy as np

from .rle import decode_rle, encode_rle

RIM = 3
MARGIN = 8


def dark_rim(image: np.ndarray, mask: np.ndarray, rim: int = RIM, margin: int = MARGIN) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if int(binary.sum()) == 0:
        return binary
    level = float(np.median(gray[binary > 0])) + margin
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim * 2 + 1, rim * 2 + 1))
    dilated = cv2.dilate(binary, kernel)
    extra = (dilated > 0) & (binary == 0) & (gray <= level)
    grown = binary.copy()
    grown[extra] = 1
    return grown


def dark_rim_rles(image: np.ndarray, rles: list[str]) -> list[str]:
    return [encode_rle(dark_rim(image, decode_rle(rle))) for rle in rles]
