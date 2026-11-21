"""Claim the image basin around one mask. The mask is the only foreground marker.

The pad is 48 pixels. This is not a peak split of the mask, not GrabCut,
and not a fixed morphological kernel.
"""

from __future__ import annotations

import cv2
import numpy as np

from .rle import decode_rle, encode_rle

PAD = 48


def marker_watershed(image: np.ndarray, mask: np.ndarray, pad: int = PAD) -> np.ndarray:
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
    color = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    markers = np.zeros(crop.shape, np.int32)
    markers[0, :] = 1
    markers[-1, :] = 1
    markers[:, 0] = 1
    markers[:, -1] = 1
    markers[binary[y0:y1, x0:x1] > 0] = 2
    cv2.watershed(color, markers)
    claimed = (markers == 2).astype(np.uint8)
    out = binary.copy()
    out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], claimed)
    return out


def refine_rles(image: np.ndarray, rles: list[str]) -> list[str]:
    return [encode_rle(marker_watershed(image, decode_rle(rle))) for rle in rles]
