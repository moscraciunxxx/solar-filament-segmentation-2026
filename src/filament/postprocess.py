from __future__ import annotations

import cv2
import numpy as np


def connected_instances(
    binary: np.ndarray,
    min_area: int = 96,
    max_area: int = 80_000,
    max_fill: float = 0.97,
    max_circularity: float = 0.92,
) -> list[np.ndarray]:
    """Split a semantic mask into instance masks, dropping speckle and sunspot-like blobs."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    instances: list[np.ndarray] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x, y, w, h = (int(stats[i, k]) for k in (
            cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT,
        ))
        bbox_area = max(w * h, 1)
        fill = area / bbox_area
        contours, _ = cv2.findContours(
            (labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        peri = cv2.arcLength(contours[0], True) if contours else 0.0
        circularity = (4.0 * np.pi * area / (peri * peri)) if peri > 0 else 0.0
        if fill > max_fill and circularity > max_circularity:
            continue
        instances.append((labels == i).astype(np.uint8))
    instances.sort(key=lambda m: int(m.sum()), reverse=True)
    return instances[:40]


def resolve_overlaps(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Greedy highest-area ownership so instances do not share pixels."""
    if not masks:
        return []
    claimed = np.zeros_like(masks[0], dtype=np.uint8)
    kept = []
    for mask in masks:
        exclusive = ((mask > 0) & (claimed == 0)).astype(np.uint8)
        if exclusive.sum() < 32:
            continue
        claimed = np.maximum(claimed, exclusive)
        kept.append(exclusive)
    return kept


def merge_nearby(masks: list[np.ndarray], gap: int = 9, max_merged_area: int = 80_000) -> list[np.ndarray]:
    """Reconnect almost-touching pieces with one dilate + relabel (linear in pixels)."""
    if len(masks) <= 1:
        return masks
    canvas = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        canvas = np.maximum(canvas, mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap))
    dilated = cv2.dilate(canvas, kernel, iterations=1)
    n, labels = cv2.connectedComponents(dilated, connectivity=8)
    merged: list[np.ndarray] = []
    for lab in range(1, n):
        region = (labels == lab)
        inst = (canvas.astype(bool) & region).astype(np.uint8)
        area = int(inst.sum())
        if area == 0 or area > max_merged_area:
            if area > max_merged_area:
                for mask in masks:
                    piece = (mask.astype(bool) & region).astype(np.uint8)
                    if piece.sum() >= 96:
                        merged.append(piece)
            continue
        merged.append(inst)
    merged.sort(key=lambda m: int(m.sum()), reverse=True)
    return merged[:40]
