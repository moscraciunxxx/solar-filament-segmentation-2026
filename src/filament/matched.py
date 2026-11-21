"""Search for a fixed dark bar at four angles, away from parent masks.

Length 61, width 5, threshold 0.55, exclusion 8 pixels. None is swept.
This is not a Hessian ridge and not a second trained model.
"""

from __future__ import annotations

import cv2
import numpy as np

from .rle import decode_rle, encode_rle
from .small_drop import MIN_AREA

LENGTH = 61
WIDTH = 5
THRESHOLD = 0.55
EXCLUDE_PX = 8
ANGLES = (0, 45, 90, 135)


def _kernel(length: int = LENGTH, width: int = WIDTH) -> np.ndarray:
    pad = 4
    kernel = np.zeros((width + 2 * pad, length), dtype=np.float32)
    kernel[pad : pad + width, :] = -1.0
    kernel[pad + width // 2, :] = -2.0
    return kernel


def _rotate(kernel: np.ndarray, angle: float) -> np.ndarray:
    center = (kernel.shape[1] / 2.0, kernel.shape[0] / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        kernel,
        matrix,
        (kernel.shape[1], kernel.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )


def match_rles(
    image: np.ndarray,
    parent_rles: list[str],
    threshold: float = THRESHOLD,
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
    gray = image.astype(np.float32)
    response = np.zeros(gray.shape, dtype=np.float32)
    base = _kernel()
    for angle in ANGLES:
        templ = _rotate(base, angle)
        matched = cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED)
        th, tw = templ.shape[:2]
        y0 = th // 2
        x0 = tw // 2
        response[y0 : y0 + matched.shape[0], x0 : x0 + matched.shape[1]] = np.maximum(
            response[y0 : y0 + matched.shape[0], x0 : x0 + matched.shape[1]],
            matched,
        )
    binary = ((response >= threshold) & (occupied == 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    rles: list[str] = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < MIN_AREA:
            continue
        rles.append(encode_rle((labels == index).astype(np.uint8)))
    return rles
