"""Expand one mask inside a fixed pad using GrabCut.

The pad is 32 pixels and GrabCut runs for 2 iterations. The original seed
is kept. No second model is added.
"""

from __future__ import annotations

import cv2
import numpy as np

from .rle import decode_rle, encode_rle

PAD = 32
ITERS = 2


def refine_mask(image: np.ndarray, mask: np.ndarray, pad: int = PAD, iters: int = ITERS) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(binary)
    if ys.size < 20:
        return binary
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, image.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, image.shape[1])
    if y1 - y0 < 8 or x1 - x0 < 8:
        return binary
    crop = image[y0:y1, x0:x1]
    color = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if crop.ndim == 2 else crop
    seed = binary[y0:y1, x0:x1]
    labels = np.full(seed.shape, cv2.GC_PR_BGD, np.uint8)
    labels[seed > 0] = cv2.GC_FGD
    labels[0, :] = cv2.GC_BGD
    labels[-1, :] = cv2.GC_BGD
    labels[:, 0] = cv2.GC_BGD
    labels[:, -1] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(color, labels, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return binary
    refined = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    out = binary.copy()
    out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], refined)
    return out


def refine_rles(image: np.ndarray, rles: list[str]) -> list[str]:
    return [encode_rle(refine_mask(image, decode_rle(rle))) for rle in rles]
