"""Split a mask only when a 2-pixel erosion breaks it into exactly two cores.

A filament body wider than that neck stays one piece. This is not the
48-pixel distance-transform watershed, which split true filaments.
"""

from __future__ import annotations

import cv2
import numpy as np

from .rle import decode_rle, encode_rle

ERODE = 2
MIN_PIECE = 32


def pinch_split(mask: np.ndarray, erode: int = ERODE) -> list[np.ndarray]:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if int(binary.sum()) < MIN_PIECE * 2:
        return [binary]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode * 2 + 1, erode * 2 + 1))
    core = cv2.erode(binary, kernel)
    count, labels = cv2.connectedComponents(core, connectivity=8)
    if count != 3:
        return [binary]
    outside_a = np.where(labels == 1, 0, 1).astype(np.uint8)
    outside_b = np.where(labels == 2, 0, 1).astype(np.uint8)
    dist_a = cv2.distanceTransform(outside_a, cv2.DIST_L2, 3)
    dist_b = cv2.distanceTransform(outside_b, cv2.DIST_L2, 3)
    on = binary > 0
    piece_a = (on & (dist_a <= dist_b)).astype(np.uint8)
    piece_b = (on & (dist_b < dist_a)).astype(np.uint8)
    if int(piece_a.sum()) < MIN_PIECE or int(piece_b.sum()) < MIN_PIECE:
        return [binary]
    return [piece_a, piece_b]


def pinch_rles(rles: list[str]) -> list[str]:
    out: list[str] = []
    for rle in rles:
        out.extend(encode_rle(piece) for piece in pinch_split(decode_rle(rle)))
    return out
