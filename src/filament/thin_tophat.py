"""Propose thin dark lanes that the parent masks do not touch.

A circular black top-hat of a fixed diameter responds to dark features
narrower than that disk and stays quiet on a wide sunspot. The image is
median-flattened first so limb darkening is not a lane. Cuts are fixed.
This is not a Hessian ridge, a CLAHE residual percentile, a bar template,
or a morphological close of an existing mask.
"""

from __future__ import annotations

import cv2
import numpy as np

from .classical import disk_median_flatten
from .preprocess import solar_disk_mask
from .rle import decode_rle, encode_rle

# Train folds 1-4, 80 images, 967 filaments: width 75th percentile 33 px,
# local contrast 25th percentile 11 gray levels. Not chosen on fold 0.
DIAMETER = 33
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


def propose_thin_masks(image: np.ndarray, parent_rles: list[str]) -> list[np.ndarray]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flat = disk_median_flatten(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DIAMETER, DIAMETER))
    blackhat = cv2.morphologyEx(flat, cv2.MORPH_BLACKHAT, kernel)
    disk = solar_disk_mask(gray)
    occupied = np.zeros(gray.shape, dtype=np.uint8)
    for rle in parent_rles:
        occupied = np.maximum(occupied, decode_rle(rle))
    if EXCLUDE_PX > 0 and int(occupied.sum()):
        block = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (EXCLUDE_PX * 2 + 1, EXCLUDE_PX * 2 + 1)
        )
        occupied = cv2.dilate(occupied, block)
    binary = (
        (blackhat >= RESPONSE_MIN) & (disk > 0) & (occupied == 0)
    ).astype(np.uint8)
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
    return kept[:40]


def propose_thin_rles(image: np.ndarray, parent_rles: list[str]) -> list[str]:
    return [encode_rle(mask) for mask in propose_thin_masks(image, parent_rles)]
