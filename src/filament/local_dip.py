"""Propose lanes that are dark relative to a 31-pixel neighborhood.

The missed filaments are about 11 gray levels below that neighborhood and
are not dark against the large-scale quiet Sun, so a disk flatten does not
see them. The window and the 11-gray cut are fixed. This is not a black
top-hat, not a CLAHE residual, and not a Hessian ridge.
"""

from __future__ import annotations

import cv2
import numpy as np

from .preprocess import solar_disk_mask
from .rle import decode_rle, encode_rle

WINDOW = 31
RESPONSE_MIN = 11
MIN_LENGTH = 40
MIN_AREA = 64
MAX_AREA = 20_000
EXCLUDE_PX = 8


def _major_length(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if ys.size < 2:
        return 0.0
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    centered = pts - pts.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    along = centered @ vectors[:, -1]
    return float(along.max() - along.min())


def propose_dip_masks(image: np.ndarray, parent_rles: list[str]) -> list[np.ndarray]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (WINDOW, WINDOW))
    dip = blurred.astype(np.int16) - gray.astype(np.int16)
    disk = solar_disk_mask(gray)
    occupied = np.zeros(gray.shape, dtype=np.uint8)
    for rle in parent_rles:
        occupied = np.maximum(occupied, decode_rle(rle))
    if EXCLUDE_PX > 0 and int(occupied.sum()):
        block = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (EXCLUDE_PX * 2 + 1, EXCLUDE_PX * 2 + 1)
        )
        occupied = cv2.dilate(occupied, block)
    binary = ((dip >= RESPONSE_MIN) & (disk > 0) & (occupied == 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    kept: list[np.ndarray] = []
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < MIN_AREA or area > MAX_AREA:
            continue
        mask = (labels == label_id).astype(np.uint8)
        if _major_length(mask) < MIN_LENGTH:
            continue
        kept.append(mask)
    kept.sort(key=lambda mask: int(mask.sum()), reverse=True)
    return kept[:80]


def propose_dip_rles(image: np.ndarray, parent_rles: list[str]) -> list[str]:
    return [encode_rle(mask) for mask in propose_dip_masks(image, parent_rles)]
