"""Enlarge one mask by a fixed factor around its centroid.

The factor is 1.15. This is an affine scale, not a morphological close,
hole fill, or GrabCut.
"""

from __future__ import annotations

import cv2
import numpy as np

from .rle import decode_rle, encode_rle

SCALE = 1.15


def scale_mask(mask: np.ndarray, scale: float = SCALE) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(binary)
    if ys.size < 10:
        return binary
    cy = float(ys.mean())
    cx = float(xs.mean())
    height, width = binary.shape
    matrix = np.array(
        [[scale, 0.0, cx - scale * cx], [0.0, scale, cy - scale * cy]],
        dtype=np.float32,
    )
    warped = cv2.warpAffine(binary, matrix, (width, height), flags=cv2.INTER_NEAREST)
    return (warped > 0).astype(np.uint8)


def scale_rles(rles: list[str], scale: float = SCALE) -> list[str]:
    return [encode_rle(scale_mask(decode_rle(rle), scale)) for rle in rles]
