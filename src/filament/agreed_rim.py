"""Add a partner's pixels only inside the locked erosion rim.

A base mask grows only where a touching partner already marks the pixel
and that pixel sits inside one 3x3 ellipse dilation of the base. A partner
that shares no pixel is ignored. Contested rim pixels go to the nearer base.
"""

from __future__ import annotations

import cv2
import numpy as np

from .yolo_export import ERODE_GT_ITERS, ERODE_GT_K


def _iou_row(mask: np.ndarray, partners: list[np.ndarray]) -> int:
    flat = mask.ravel().astype(np.float32)
    area = float(flat.sum())
    best = -1
    best_iou = 0.0
    for index, partner in enumerate(partners):
        other = partner.ravel().astype(np.float32)
        inter = float(flat @ other)
        if inter <= 0.0:
            continue
        union = area + float(other.sum()) - inter
        iou = inter / union if union > 0.0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best = index
    return best


def agreed_rim_masks(
    bases: list[np.ndarray],
    partners: list[np.ndarray],
    k: int = ERODE_GT_K,
    iters: int = ERODE_GT_ITERS,
) -> list[np.ndarray]:
    if not bases:
        return []
    shape = bases[0].shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    grown: list[np.ndarray] = []
    for mask in bases:
        core = (mask > 0).astype(np.uint8)
        if partners:
            partner_index = _iou_row(core, partners)
        else:
            partner_index = -1
        if partner_index < 0:
            grown.append(core)
            continue
        zone = cv2.dilate(core, kernel, iterations=iters)
        extra = (zone > 0) & (partners[partner_index] > 0) & (core == 0)
        grown.append((core | extra).astype(np.uint8))
    if len(grown) == 1:
        return grown
    dist = np.full((len(grown),) + shape, np.inf, dtype=np.float32)
    for index, mask in enumerate(bases):
        core = (mask > 0).astype(np.uint8)
        if int(core.sum()) == 0:
            continue
        inv = np.where(core > 0, np.uint8(0), np.uint8(1))
        local = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        dist[index] = np.where(grown[index] > 0, local, np.inf)
    winner = np.argmin(dist, axis=0)
    finite = np.isfinite(dist).any(axis=0)
    return [((winner == index) & finite).astype(np.uint8) for index in range(len(grown))]
