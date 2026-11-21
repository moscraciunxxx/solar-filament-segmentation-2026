"""Propose filaments by correlation with real training-fold miss patches.

Templates are oriented crops of filaments a model missed. The keep-cut is
the higher of the holdout positive 10th percentile and the quiet-Sun 95th
percentile. This is not the fixed 61-by-5 bar at 0.55.
"""

from __future__ import annotations

import cv2
import numpy as np

from .disjoint_add import mask_iou
from .rle import decode_rle, encode_rle

PATCH_W = 64
PATCH_H = 32
ANGLES = (0, 45, 90, 135)
MIN_DISTANCE = 48
MAX_PER_IMAGE = 40


def bank_threshold(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """One cut from holdout scores. Positives at p10, quiet Sun at p95."""
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("threshold needs both positive and quiet scores")
    return float(max(np.quantile(pos, 0.10), np.quantile(neg, 0.95)))


def rasterize_label(text: str, size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        coords = np.asarray(parts[1:], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= size
        coords[:, 1] *= size
        cv2.fillPoly(mask, [np.round(coords).astype(np.int32)], 1)
    return mask


def _major_angle(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    centered = pts - pts.mean(axis=0)
    cov = centered.T @ centered
    values, vectors = np.linalg.eigh(cov)
    major = vectors[:, int(np.argmax(values))]
    return float(np.degrees(np.arctan2(major[1], major[0])))


def _rotate_pair(image: np.ndarray, mask: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(round(height * sin + width * cos))
    new_h = int(round(height * cos + width * sin))
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    fill = float(np.median(image))
    rotated = cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=fill)
    rotated_mask = cv2.warpAffine(mask, matrix, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated, (rotated_mask > 0).astype(np.uint8)


def oriented_patch(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a 64x32 horizontal patch and its mask, or None if the mask is tiny."""
    if int(mask.sum()) < 20:
        return None
    rotated, rotated_mask = _rotate_pair(image.astype(np.float32), mask, -_major_angle(mask))
    ys, xs = np.nonzero(rotated_mask)
    if xs.size < 20:
        return None
    y0 = max(int(ys.min()) - 2, 0)
    y1 = min(int(ys.max()) + 3, rotated.shape[0])
    x0 = max(int(xs.min()) - 2, 0)
    x1 = min(int(xs.max()) + 3, rotated.shape[1])
    if y1 - y0 < 3 or x1 - x0 < 3:
        return None
    patch = cv2.resize(rotated[y0:y1, x0:x1], (PATCH_W, PATCH_H), interpolation=cv2.INTER_AREA)
    patch_mask = cv2.resize(rotated_mask[y0:y1, x0:x1], (PATCH_W, PATCH_H), interpolation=cv2.INTER_NEAREST)
    return patch.astype(np.float32), (patch_mask > 0).astype(np.uint8)


def expand_angles(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    bank = []
    for angle in ANGLES:
        turned, turned_mask = _rotate_pair(patch, mask, angle)
        if int(turned_mask.sum()) < 10:
            continue
        if turned.shape[0] < 8 or turned.shape[1] < 8:
            continue
        bank.append((turned.astype(np.float32), turned_mask))
    return bank


def max_correlation(image: np.ndarray, bank: list[tuple[np.ndarray, np.ndarray]]) -> float:
    gray = image.astype(np.float32)
    best = -1.0
    for patch, _mask in bank:
        if patch.shape[0] >= gray.shape[0] or patch.shape[1] >= gray.shape[1]:
            continue
        matched = cv2.matchTemplate(gray, patch, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(np.max(matched)))
    return best


def propose_template_masks(
    image: np.ndarray,
    bank: list[tuple[np.ndarray, np.ndarray]],
    threshold: float,
    occupied: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Paste template masks at correlation peaks that clear ``threshold``."""
    gray = image.astype(np.float32)
    scored: list[tuple[float, np.ndarray]] = []
    kernel = np.ones((MIN_DISTANCE, MIN_DISTANCE), dtype=np.uint8)
    for patch, mask in bank:
        if patch.shape[0] >= gray.shape[0] or patch.shape[1] >= gray.shape[1]:
            continue
        matched = cv2.matchTemplate(gray, patch, cv2.TM_CCOEFF_NORMED)
        local_max = cv2.dilate(matched, kernel)
        peaks = np.argwhere((matched >= threshold) & (matched >= local_max - 1e-6))
        height, width = mask.shape
        for y, x in peaks:
            full = np.zeros(gray.shape, dtype=np.uint8)
            full[int(y) : int(y) + height, int(x) : int(x) + width] = mask
            if int(full.sum()) < 10:
                continue
            scored.append((float(matched[int(y), int(x)]), full))
    scored.sort(key=lambda item: -item[0])
    kept: list[np.ndarray] = []
    centers: list[tuple[float, float]] = []
    for score, mask in scored:
        if occupied is not None and np.any((mask > 0) & (occupied > 0)):
            continue
        ys, xs = np.nonzero(mask)
        cx = float(xs.mean())
        cy = float(ys.mean())
        if any((cx - ox) ** 2 + (cy - oy) ** 2 < MIN_DISTANCE ** 2 for ox, oy in centers):
            continue
        kept.append(mask)
        centers.append((cx, cy))
        if len(kept) >= MAX_PER_IMAGE:
            break
    return kept


def propose_template_rles(
    image: np.ndarray,
    parent_rles: list[str],
    bank: list[tuple[np.ndarray, np.ndarray]],
    threshold: float,
) -> list[str]:
    occupied = np.zeros(image.shape[:2], dtype=np.uint8)
    parent_masks = []
    for rle in parent_rles:
        mask = decode_rle(rle, image.shape[0], image.shape[1])
        parent_masks.append(mask)
        occupied = np.maximum(occupied, mask)
    proposed = propose_template_masks(image, bank, threshold, occupied)
    rles = []
    for mask in proposed:
        if any(mask_iou(mask, other) != 0.0 for other in parent_masks):
            continue
        rles.append(encode_rle(mask))
    return rles
