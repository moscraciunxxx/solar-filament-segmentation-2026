"""Dark-line response from the Hessian. Fixed scale, not a model paste.

Sigma is 3 pixels. A component must be at least 80 pixels long and 96
pixels in area, and it must sit 8 pixels off every parent mask.
"""

from __future__ import annotations

import cv2
import numpy as np

from .empty_region import _major_length
from .rle import decode_rle, encode_rle
from .small_drop import MIN_AREA

SIGMA = 3.0
MIN_LENGTH = 80
EXCLUDE_PX = 8
RIDGE_PERCENTILE = 99.5


def dark_ridge(image: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    blur = cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    hxx = cv2.Sobel(gx, cv2.CV_32F, 1, 0, ksize=3)
    hyy = cv2.Sobel(gy, cv2.CV_32F, 0, 1, ksize=3)
    hxy = cv2.Sobel(gx, cv2.CV_32F, 0, 1, ksize=3)
    gap = np.sqrt((hxx - hyy) ** 2 + 4.0 * hxy ** 2)
    lam1 = 0.5 * (hxx + hyy - gap)
    lam2 = 0.5 * (hxx + hyy + gap)
    return np.maximum(lam1, lam2)


def ridge_rles(
    image: np.ndarray,
    parent_rles: list[str],
    sigma: float = SIGMA,
    min_length: int = MIN_LENGTH,
    exclude_px: int = EXCLUDE_PX,
) -> list[str]:
    occupied = np.zeros(image.shape[:2], dtype=np.uint8)
    for rle in parent_rles:
        occupied = np.maximum(occupied, decode_rle(rle))
    if exclude_px > 0 and int(occupied.sum()):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (exclude_px * 2 + 1, exclude_px * 2 + 1)
        )
        occupied = cv2.dilate(occupied, kernel)
    response = dark_ridge(image, sigma)
    outside = occupied == 0
    if int(outside.sum()) == 0:
        return []
    threshold = float(np.percentile(response[outside], RIDGE_PERCENTILE))
    binary = ((response >= threshold) & outside).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    rles: list[str] = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < MIN_AREA:
            continue
        mask = (labels == index).astype(np.uint8)
        if _major_length(mask) < min_length:
            continue
        rles.append(encode_rle(mask))
    return rles
