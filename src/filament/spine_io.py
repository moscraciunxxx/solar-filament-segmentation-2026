"""Train-JSON spine polylines. Never load test / Dataverse labels.

v1 U-Net decode does not import this. v2 watershed only, after merge HOLD.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .paths import IMAGE_SIZE, TRAIN_JSON


def spine_to_xy(spine) -> np.ndarray:
    if spine is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(spine, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    return arr.reshape(-1, 2)


def rasterize_spine_ridge(pts: np.ndarray, size: int = IMAGE_SIZE, width: int = 1) -> np.ndarray:
    ridge = np.zeros((size, size), dtype=np.uint8)
    if len(pts) < 2:
        return ridge
    xy = np.round(pts).astype(np.int32)
    xy[:, 0] = np.clip(xy[:, 0], 0, size - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, size - 1)
    cv2.polylines(ridge, [xy], isClosed=False, color=1, thickness=max(1, width))
    return ridge


def load_spines_by_file(path: Path = TRAIN_JSON) -> dict[str, list[np.ndarray]]:
    """file_name -> list of spine polylines from every annotator on that JPEG."""
    with path.open() as f:
        coco = json.load(f)
    images = {img["id"]: img["file_name"] for img in coco["images"]}
    by_file: dict[str, list[np.ndarray]] = defaultdict(list)
    for ann in coco["annotations"]:
        pts = spine_to_xy(ann.get("spine"))
        if len(pts) >= 2:
            by_file[images[ann["image_id"]]].append(pts)
    return dict(by_file)


def union_ridge_for_file(
    file_name: str,
    spines_by_file: dict[str, list[np.ndarray]],
    size: int = IMAGE_SIZE,
    width: int = 1,
) -> np.ndarray:
    ridge = np.zeros((size, size), dtype=np.uint8)
    for pts in spines_by_file.get(file_name, []):
        ridge = np.maximum(ridge, rasterize_spine_ridge(pts, size=size, width=width))
    return ridge
