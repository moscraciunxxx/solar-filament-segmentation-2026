"""Native-2048 1024-crop boxes and probability-map stitch.

Geometry helper for fail-path rank 2 `unet_patch_native_1024_dhebe`.
Do not import from scripts/eval_unet.py. No image or label I/O.
Do not reuse YOLO tile export (GT-centric 768 crops + instance cluster).
"""

from __future__ import annotations

import numpy as np

from .paths import IMAGE_SIZE

PATCH_SIZE = 1024
PATCH_OVERLAP = 256


def patch_axis_origins(
    image_size: int = IMAGE_SIZE,
    patch: int = PATCH_SIZE,
    overlap: int = PATCH_OVERLAP,
) -> list[int]:
    """Flush the last crop so `image_size` is covered on one axis."""
    if patch <= overlap or patch > image_size:
        raise ValueError(
            f"Need 0 <= overlap < patch <= image_size, got patch={patch} overlap={overlap}"
        )
    stride = patch - overlap
    last = image_size - patch
    xs = list(range(0, last + 1, stride))
    if xs[-1] != last:
        xs.append(last)
    return xs


def patch_boxes(
    image_size: int = IMAGE_SIZE,
    patch: int = PATCH_SIZE,
    overlap: int = PATCH_OVERLAP,
) -> list[tuple[int, int, int, int]]:
    """1024-crop boxes on a 2048 canvas as (y0, x0, y1, x1), exclusive ends, row-major."""
    origins = patch_axis_origins(image_size, patch, overlap)
    boxes: list[tuple[int, int, int, int]] = []
    for y0 in origins:
        for x0 in origins:
            boxes.append((y0, x0, y0 + patch, x0 + patch))
    return boxes


def crop_patch(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    y0, x0, y1, x1 = box
    return array[y0:y1, x0:x1]


def stitch_max(
    patches: list[np.ndarray],
    boxes: list[tuple[int, int, int, int]],
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Overlap keeps the stronger probability so 1–3 px threads survive the seam."""
    if len(patches) != len(boxes):
        raise ValueError(f"patches ({len(patches)}) and boxes ({len(boxes)}) length mismatch")
    canvas = np.zeros((image_size, image_size), dtype=np.float32)
    for patch, (y0, x0, y1, x1) in zip(patches, boxes):
        crop = np.asarray(patch, dtype=np.float32)
        if crop.shape != (y1 - y0, x1 - x0):
            raise ValueError(f"patch shape {crop.shape} != box {(y1 - y0, x1 - x0)}")
        canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], crop)
    return canvas


def stitch_mean(
    patches: list[np.ndarray],
    boxes: list[tuple[int, int, int, int]],
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Overlap average. Ablation only — mean can dilute the 1–3 px threads stitch-max keeps."""
    if len(patches) != len(boxes):
        raise ValueError(f"patches ({len(patches)}) and boxes ({len(boxes)}) length mismatch")
    acc = np.zeros((image_size, image_size), dtype=np.float32)
    count = np.zeros((image_size, image_size), dtype=np.float32)
    for patch, (y0, x0, y1, x1) in zip(patches, boxes):
        crop = np.asarray(patch, dtype=np.float32)
        if crop.shape != (y1 - y0, x1 - x0):
            raise ValueError(f"patch shape {crop.shape} != box {(y1 - y0, x1 - x0)}")
        acc[y0:y1, x0:x1] += crop
        count[y0:y1, x0:x1] += 1.0
    return np.divide(acc, count, out=np.zeros_like(acc), where=count > 0)
