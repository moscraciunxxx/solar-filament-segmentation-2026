"""Dark components in a fixed annulus inside the solar limb.

The disk radius is 1000. The ring runs from 880 to 980 pixels from center.
Components must be at least 96 pixels in area and 40 pixels long, and 8
pixels off every parent mask. None of those cuts is swept.
"""

from __future__ import annotations

import cv2
import numpy as np

from .empty_region import _major_length
from .rle import decode_rle, encode_rle
from .small_drop import MIN_AREA

DISK_RADIUS = 1000.0
INNER = 880.0
OUTER = 980.0
MIN_LENGTH = 40
EXCLUDE_PX = 8
DARK_FRACTION = 0.10


def limb_annulus(shape: tuple[int, int], inner: float = INNER, outer: float = OUTER) -> np.ndarray:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return ((dist >= inner) & (dist <= outer)).astype(np.uint8)


def limb_rles(image: np.ndarray, parent_rles: list[str]) -> list[str]:
    ring = limb_annulus(image.shape[:2])
    occupied = np.zeros(image.shape[:2], dtype=np.uint8)
    for rle in parent_rles:
        occupied = np.maximum(occupied, decode_rle(rle))
    if int(occupied.sum()):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (EXCLUDE_PX * 2 + 1, EXCLUDE_PX * 2 + 1)
        )
        occupied = cv2.dilate(occupied, kernel)
    region = (ring > 0) & (occupied == 0)
    if int(region.sum()) == 0:
        return []
    cutoff = float(np.quantile(image[region], DARK_FRACTION))
    binary = (region & (image <= cutoff)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    rles: list[str] = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < MIN_AREA:
            continue
        mask = (labels == index).astype(np.uint8)
        if _major_length(mask) < MIN_LENGTH:
            continue
        rles.append(encode_rle(mask))
    return rles
