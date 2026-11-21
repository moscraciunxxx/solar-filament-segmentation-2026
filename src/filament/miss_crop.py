"""Crop window around a filament the parent model missed.

Fold 0 is not a source of these crops. The window is 640 pixels and stays
inside the 2048 frame. A polygon that misses the window produces no label line.
"""

from __future__ import annotations

import numpy as np

from .yolo_export import polygon_to_yolo_line

CROP = 640
MISS_IOU = 0.5
STRIDE = 320
SPECIALIST_CONF = 0.50
DISK_CENTER = 1023.5
DISK_KEEP = 680.0
NEG_PER_IMAGE = 2


def window_origins(
    image_size: int = 2048,
    crop: int = CROP,
    stride: int = STRIDE,
) -> list[int]:
    """Window starts that cover the frame, including the far edge."""
    if crop > image_size or stride <= 0:
        raise ValueError("window does not fit the frame")
    origins = list(range(0, image_size - crop + 1, stride))
    last = image_size - crop
    if not origins or origins[-1] != last:
        origins.append(last)
    return origins


def window_is_quiet(union: np.ndarray, x0: int, y0: int, crop: int = CROP) -> bool:
    """True when the window misses every filament pixel and sits on the disk."""
    if union.ndim != 2:
        raise ValueError("union is a mask")
    y1 = y0 + crop
    x1 = x0 + crop
    if y0 < 0 or x0 < 0 or y1 > union.shape[0] or x1 > union.shape[1]:
        return False
    if int(union[y0:y1, x0:x1].sum()) != 0:
        return False
    cx = x0 + crop / 2.0
    cy = y0 + crop / 2.0
    dx = cx - DISK_CENTER
    dy = cy - DISK_CENTER
    return dx * dx + dy * dy <= DISK_KEEP * DISK_KEEP


def quiet_origins(
    union: np.ndarray,
    rng: np.random.Generator,
    count: int = NEG_PER_IMAGE,
) -> list[tuple[int, int]]:
    """Up to ``count`` on-disk windows that contain no filament pixel."""
    origins = window_origins(union.shape[0], CROP, STRIDE)
    candidates = [
        (x0, y0)
        for y0 in origins
        for x0 in origins
        if window_is_quiet(union, x0, y0)
    ]
    if not candidates or count <= 0:
        return []
    order = rng.permutation(len(candidates))
    chosen = [candidates[int(i)] for i in order[:count]]
    return chosen


def crop_origin(
    cx: float,
    cy: float,
    image_size: int = 2048,
    crop: int = CROP,
) -> tuple[int, int]:
    """Top-left of a crop that contains the centroid and stays on the frame."""
    if crop > image_size:
        raise ValueError("crop is larger than the frame")
    half = crop // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x0 = min(max(x0, 0), image_size - crop)
    y0 = min(max(y0, 0), image_size - crop)
    return x0, y0


def polygon_in_crop(
    polygon: np.ndarray,
    x0: int,
    y0: int,
    crop: int = CROP,
) -> str | None:
    """YOLO line for a polygon shifted into the crop, or None if it misses it."""
    shifted = np.asarray(polygon, dtype=np.float32).copy()
    if shifted.ndim != 2 or shifted.shape[0] < 3 or shifted.shape[1] != 2:
        return None
    shifted[:, 0] -= float(x0)
    shifted[:, 1] -= float(y0)
    if (
        float(shifted[:, 0].max()) < 0.0
        or float(shifted[:, 1].max()) < 0.0
        or float(shifted[:, 0].min()) >= crop
        or float(shifted[:, 1].min()) >= crop
    ):
        return None
    return polygon_to_yolo_line(shifted, size=crop)
