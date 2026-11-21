"""Fill interior holes of one mask. The outer boundary is unchanged."""

from __future__ import annotations

import cv2
import numpy as np

from .rle import decode_rle, encode_rle


def fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape
    flood = binary.copy()
    guard = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flood, guard, (0, 0), 1)
    holes = flood == 0
    filled = binary.copy()
    filled[holes] = 1
    return filled


def fill_rles(rles: list[str]) -> list[str]:
    return [encode_rle(fill_holes(decode_rle(rle))) for rle in rles]
