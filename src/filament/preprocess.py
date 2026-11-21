from __future__ import annotations

import cv2
import numpy as np

from .paths import IMAGE_SIZE


def read_gray(path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return image


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(image)


def solar_disk_mask(image: np.ndarray, intensity_min: int = 12, radius: int = 1000) -> np.ndarray:
    """Conservative on-disk mask: bright enough and inside a centered circle."""
    h, w = image.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return ((image > intensity_min) & (dist <= radius)).astype(np.uint8)
