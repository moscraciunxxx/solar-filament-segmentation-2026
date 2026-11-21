"""Drop round instance masks. Filaments are elongated; sunspots are not.

The cut is fixed before any Panoptic Quality is computed: keep a mask when
its area is below ``min_area`` (too small to judge) or its inertia
eccentricity is at least ``min_eccentricity``. This does not add instances,
bridge neighbors, or edit boundaries.
"""

from __future__ import annotations

import cv2
import numpy as np

from .rle import decode_rle

MIN_ECCENTRICITY = 0.90
MIN_AREA = 64


def mask_eccentricity(mask: np.ndarray) -> float:
    moments = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    area = float(moments["m00"])
    if area <= 0.0:
        return 0.0
    mu20 = moments["mu20"] / area
    mu02 = moments["mu02"] / area
    mu11 = moments["mu11"] / area
    trace = mu20 + mu02
    det = mu20 * mu02 - mu11 * mu11
    disc = max(trace * trace - 4.0 * det, 0.0)
    root = disc ** 0.5
    major = 0.5 * (trace + root)
    minor = 0.5 * (trace - root)
    if major <= 1e-9:
        return 0.0
    ratio = max(minor, 0.0) / major
    return float((1.0 - min(ratio, 1.0)) ** 0.5)


def keep_elongated(
    mask: np.ndarray,
    min_eccentricity: float = MIN_ECCENTRICITY,
    min_area: int = MIN_AREA,
) -> bool:
    area = int((mask > 0).sum())
    if area < min_area:
        return True
    return mask_eccentricity(mask) >= min_eccentricity


def filter_rles(
    rles: list[str],
    min_eccentricity: float = MIN_ECCENTRICITY,
    min_area: int = MIN_AREA,
) -> list[str]:
    kept: list[str] = []
    for rle in rles:
        mask = decode_rle(rle)
        if keep_elongated(mask, min_eccentricity, min_area):
            kept.append(rle)
    return kept
