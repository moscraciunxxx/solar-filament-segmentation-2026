"""Center-to-limb flat field. One channel. No CLAHE, Hessian, or z-score."""

from __future__ import annotations

import cv2
import numpy as np


def center_to_limb_flat(gray: np.ndarray, bins: int = 32) -> np.ndarray:
    """Ray-cast the limb, divide by the annular profile and a slow background.

    Returns a float32 image in [0, 1]. Off-disk pixels are 0.
    """
    image = gray.astype(np.float32)
    if float(image.max()) > 1.5:
        image = image / 255.0
    height, width = image.shape
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    yy, xx = np.mgrid[0:height, 0:width]
    rho = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = int(min(cy, cx, height - 1 - cy, width - 1 - cx))
    r_max = max(r_max, 2)
    r_int = np.clip(rho.astype(np.int32), 0, r_max - 1)
    sums = np.bincount(r_int.ravel(), weights=image.ravel(), minlength=r_max)
    counts = np.bincount(r_int.ravel(), minlength=r_max).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    profile = sums / counts
    outer = int(r_max * 0.55)
    drop = np.diff(profile)
    limb = outer + int(np.argmin(drop[outer:])) if outer < len(drop) else r_max - 1
    limb = max(int(limb), 1)
    inside = rho <= limb
    bin_id = np.clip((rho / limb * bins).astype(np.int32), 0, bins - 1)
    clv = np.ones(bins, dtype=np.float32)
    for index in range(bins):
        sample = image[(bin_id == index) & inside]
        if sample.size:
            clv[index] = float(np.median(sample))
    background = cv2.GaussianBlur(image, (0, 0), sigmaX=max(limb / 10.0, 1.0))
    denom = np.maximum(clv[bin_id] * np.maximum(background, 1e-3), 1e-3)
    flat = np.clip(image / denom, 0.0, 2.0) / 2.0
    flat = np.where(inside, flat, 0.0)
    return flat.astype(np.float32)
