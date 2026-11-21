"""Extend a mask a fixed distance along its major axis only.

The outer sides are not dilated. The distance is fixed at 8 pixels.
"""

from __future__ import annotations

import numpy as np

from .rle import decode_rle, encode_rle

TIP_PX = 8


def extend_tips(mask: np.ndarray, extra: int = TIP_PX) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(binary)
    if ys.size < 32:
        return binary
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    center = pts.mean(axis=0)
    centered = pts - center
    _, vectors = np.linalg.eigh(centered.T @ centered)
    major = vectors[:, -1]
    minor = vectors[:, 0]
    along = centered @ major
    across = centered @ minor
    half_width = max(float(np.std(across)), 1.0)
    lo = float(along.min()) - extra
    hi = float(along.max()) + extra
    x0 = max(int(xs.min()) - extra, 0)
    x1 = min(int(xs.max()) + extra + 1, binary.shape[1])
    y0 = max(int(ys.min()) - extra, 0)
    y1 = min(int(ys.max()) + extra + 1, binary.shape[0])
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rel_x = xx.astype(np.float64) - center[0]
    rel_y = yy.astype(np.float64) - center[1]
    proj = rel_x * major[0] + rel_y * major[1]
    side = rel_x * minor[0] + rel_y * minor[1]
    band = (proj >= lo) & (proj <= hi) & (np.abs(side) <= half_width)
    out = binary.copy()
    out[y0:y1, x0:x1][band] = 1
    return out


def extend_tips_inside(
    mask: np.ndarray,
    partner: np.ndarray,
    extra: int = TIP_PX,
) -> np.ndarray:
    """Add major-axis tip pixels only where the partner already marks them.

    A partner that shares no pixel is ignored. Pixels the extension would
    paint on empty disk stay empty. The distance stays at 8.
    """
    binary = (mask > 0).astype(np.uint8)
    cover = partner > 0
    if not np.any(binary & cover):
        return binary
    extended = extend_tips(binary, extra)
    out = binary.copy()
    out[(extended > 0) & (binary == 0) & cover] = 1
    return out


def extend_rles(rles: list[str], extra: int = TIP_PX) -> list[str]:
    return [encode_rle(extend_tips(decode_rle(rle), extra)) for rle in rles]
