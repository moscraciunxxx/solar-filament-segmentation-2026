"""Propose a dark lane whose edges point the same way.

The neighborhood is 96 pixels, three times the killed 31-pixel dip, and a
pixel must also have structure-tensor coherence of at least 0.40. Area and
darkness cuts are supplied by the caller from folds other than 0. This does
not edit an existing mask and it does not rerun the 31-pixel dip.
"""

from __future__ import annotations

import cv2
import numpy as np

from .preprocess import solar_disk_mask
from .rle import decode_rle, encode_rle

WINDOW = 96
COHERENCE_MIN = 0.40
FAR_PX = 64
MIN_LENGTH = 40


def _major_length(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if ys.size < 2:
        return 0.0
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    centered = pts - pts.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    along = centered @ vectors[:, -1]
    return float(along.max() - along.min())


def edge_coherence(gray: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """How strongly the local gradient points along one axis. Range 0 to 1."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    trace = jxx + jyy
    gap = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    major = 0.5 * (trace + gap)
    minor = 0.5 * (trace - gap)
    return (major - minor) / (major + minor + 1e-6)


def propose_coherent_masks(
    image: np.ndarray,
    occupied: np.ndarray,
    *,
    response_min: float,
    min_area: int,
    max_area: int,
    window: int = WINDOW,
    coherence_min: float = COHERENCE_MIN,
    far_px: int = FAR_PX,
    min_length: int = MIN_LENGTH,
) -> list[np.ndarray]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if window % 2 == 0:
        window += 1
    blurred = cv2.blur(gray, (window, window))
    dip = blurred.astype(np.float32) - gray.astype(np.float32)
    coherence = edge_coherence(gray)
    disk = solar_disk_mask(gray)
    blocked = (occupied > 0).astype(np.uint8)
    if far_px > 0 and int(blocked.sum()):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (far_px * 2 + 1, far_px * 2 + 1))
        blocked = cv2.dilate(blocked, kernel)
    binary = (
        (dip >= response_min) & (coherence >= coherence_min) & (disk > 0) & (blocked == 0)
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    kept: list[np.ndarray] = []
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        mask = (labels == label_id).astype(np.uint8)
        if _major_length(mask) < min_length:
            continue
        kept.append(mask)
    return kept


def propose_coherent_rles(
    image: np.ndarray,
    parent_rles: list[str],
    *,
    response_min: float,
    min_area: int,
    max_area: int,
) -> list[str]:
    occupied = np.zeros(image.shape[:2], dtype=np.uint8)
    for rle in parent_rles:
        occupied = np.maximum(occupied, decode_rle(rle, image.shape[0], image.shape[1]))
    masks = propose_coherent_masks(
        image,
        occupied,
        response_min=response_min,
        min_area=min_area,
        max_area=max_area,
    )
    return [encode_rle(mask) for mask in masks]
