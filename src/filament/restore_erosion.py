"""Put back the one-pixel rim the locked training erosion removed.

Each mask is dilated with the same 3x3 ellipse used to erode the training
labels. A rim pixel claimed by two masks goes to the nearer core. Cores
stay with the mask that already owned them.
"""

from __future__ import annotations

import cv2
import numpy as np

from .yolo_export import ERODE_GT_ITERS, ERODE_GT_K


def restore_eroded_masks(
    masks: list[np.ndarray],
    k: int = ERODE_GT_K,
    iters: int = ERODE_GT_ITERS,
) -> list[np.ndarray]:
    if not masks:
        return []
    shape = masks[0].shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated: list[np.ndarray] = []
    active: list[bool] = []
    for mask in masks:
        core = (mask > 0).astype(np.uint8)
        if int(core.sum()) == 0:
            dilated.append(np.zeros(shape, dtype=np.uint8))
            active.append(False)
            continue
        dilated.append(cv2.dilate(core, kernel, iterations=iters))
        active.append(True)
    if not any(active):
        return [np.zeros(shape, dtype=np.uint8) for _ in masks]
    dist = np.full((len(masks),) + shape, np.inf, dtype=np.float32)
    for index, on in enumerate(active):
        if not on:
            continue
        inv = np.where(masks[index] > 0, np.uint8(0), np.uint8(1))
        local = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        dist[index] = np.where(dilated[index] > 0, local, np.inf)
    winner = np.argmin(dist, axis=0)
    finite = np.isfinite(dist).any(axis=0)
    return [((winner == index) & finite).astype(np.uint8) for index in range(len(masks))]
