"""Split one parent mask when it contains two all-annotator instances.

An instance counts when at least half its pixels lie inside the parent and
its area is at least 96. The two instances must not overlap each other above
0.5. Parent pixels are given to the nearer instance. This is not a neck
erosion and not a pixel vote.
"""

from __future__ import annotations

import cv2
import numpy as np

from .disjoint_add import mask_iou
from .rle import decode_rle, encode_rle
from .small_drop import MIN_AREA

CONTAIN = 0.5
MIN_PIECE = 32


def _contained(parent: np.ndarray, others: list[np.ndarray]) -> list[np.ndarray]:
    hits: list[np.ndarray] = []
    for other in others:
        area = int(other.sum())
        if area < MIN_AREA:
            continue
        inside = int(np.count_nonzero((parent > 0) & (other > 0)))
        if inside / area >= CONTAIN:
            hits.append(other)
    return hits


def contain_split(parent: np.ndarray, others: list[np.ndarray]) -> list[np.ndarray]:
    binary = (np.asarray(parent) > 0).astype(np.uint8)
    hits = _contained(binary, [(np.asarray(other) > 0).astype(np.uint8) for other in others])
    if len(hits) != 2 or mask_iou(hits[0], hits[1]) > CONTAIN:
        return [binary]
    outside_a = np.where(hits[0] > 0, 0, 1).astype(np.uint8)
    outside_b = np.where(hits[1] > 0, 0, 1).astype(np.uint8)
    dist_a = cv2.distanceTransform(outside_a, cv2.DIST_L2, 3)
    dist_b = cv2.distanceTransform(outside_b, cv2.DIST_L2, 3)
    on = binary > 0
    piece_a = (on & (dist_a <= dist_b)).astype(np.uint8)
    piece_b = (on & (dist_b < dist_a)).astype(np.uint8)
    if int(piece_a.sum()) < MIN_PIECE or int(piece_b.sum()) < MIN_PIECE:
        return [binary]
    return [piece_a, piece_b]


def contain_rles(parent_rles: list[str], other_rles: list[str]) -> tuple[list[str], int]:
    others = [decode_rle(rle) for rle in other_rles]
    out: list[str] = []
    splits = 0
    for rle in parent_rles:
        pieces = contain_split(decode_rle(rle), others)
        if len(pieces) > 1:
            splits += 1
        out.extend(encode_rle(piece) for piece in pieces)
    return out, splits
