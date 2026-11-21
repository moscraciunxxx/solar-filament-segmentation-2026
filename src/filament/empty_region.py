"""Propose a long component only in pixels away from existing masks.

The exclusion radius is 16 pixels, the minimum length is 40, and the dark
cut is the 99th percentile of the residual outside the parent. None is swept.
This is not a full-disk classical detector.
"""

from __future__ import annotations

import cv2
import numpy as np

from .classical import dark_residual_channel
from .postprocess import connected_instances
from .rle import decode_rle, encode_rle

EXCLUDE_PX = 16
MIN_LENGTH = 40
DARK_PERCENTILE = 99.0


def _major_length(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if ys.size < 2:
        return 0.0
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    centered = pts - pts.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    along = centered @ vectors[:, -1]
    return float(along.max() - along.min())


def components_far_from(
    binary: np.ndarray,
    occupied: np.ndarray,
    exclude_px: int = EXCLUDE_PX,
    min_length: int = MIN_LENGTH,
) -> list[np.ndarray]:
    blocked = (occupied > 0).astype(np.uint8)
    if exclude_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (exclude_px * 2 + 1, exclude_px * 2 + 1)
        )
        blocked = cv2.dilate(blocked, kernel)
    allowed = ((binary > 0) & (blocked == 0)).astype(np.uint8)
    kept: list[np.ndarray] = []
    for inst in connected_instances(allowed):
        if _major_length(inst) >= min_length:
            kept.append(inst)
    return kept


def propose_far_rles(image: np.ndarray, parent_rles: list[str]) -> list[str]:
    occupied = np.zeros(image.shape[:2], dtype=np.uint8)
    for rle in parent_rles:
        occupied = np.maximum(occupied, decode_rle(rle))
    residual = dark_residual_channel(image)
    outside = occupied == 0
    if int(outside.sum()) == 0:
        return []
    threshold = float(np.percentile(residual[outside], DARK_PERCENTILE))
    binary = ((residual >= threshold) & outside).astype(np.uint8)
    return [encode_rle(mask) for mask in components_far_from(binary, occupied)]
