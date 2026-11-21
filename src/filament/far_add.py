"""Add a donor mask only when it sits far from every parent mask.

Every parent mask is kept. A donor mask is appended only when it shares no
pixel with a parent mask and its centroid is at least ``FAR_PX`` from the
nearest parent pixel. Blank parent images stay blank. The distance is fixed.
"""

from __future__ import annotations

import cv2
import numpy as np

from .disjoint_add import mask_iou
from .paths import IMAGE_SIZE
from .rle import decode_rle

FAR_PX = 64


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return -1.0, -1.0
    return float(xs.mean()), float(ys.mean())


def _distance_to_parent(union: np.ndarray) -> np.ndarray:
    obstacles = np.where(union > 0, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(obstacles, cv2.DIST_L2, 5)


def far_add(
    parent_rles: list[str],
    candidate_rles: list[str],
    far_px: float = FAR_PX,
    image_size: int = IMAGE_SIZE,
) -> tuple[list[str], int]:
    """Return parent RLEs plus far, pixel-disjoint candidates."""
    if not parent_rles or not candidate_rles:
        return list(parent_rles), 0
    parent_masks = [decode_rle(rle, image_size, image_size) for rle in parent_rles]
    union = np.zeros((image_size, image_size), dtype=np.uint8)
    for mask in parent_masks:
        union = np.maximum(union, mask)
    dist = _distance_to_parent(union)
    added: list[str] = []
    for rle in candidate_rles:
        mask = decode_rle(rle, image_size, image_size)
        if any(mask_iou(mask, other) != 0.0 for other in parent_masks):
            continue
        cx, cy = _centroid(mask)
        if cx < 0.0:
            continue
        x = int(np.clip(round(cx), 0, dist.shape[1] - 1))
        y = int(np.clip(round(cy), 0, dist.shape[0] - 1))
        if float(dist[y, x]) >= far_px:
            added.append(rle)
    return list(parent_rles) + added, len(added)
