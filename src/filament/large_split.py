"""Split only a mask larger than the train area cap.

Two or more all-annotator instances must each lie at least half inside it.
Smaller masks stay whole. This does not delete the large mask and does not
edit a boundary pixel by brightness.
"""

from __future__ import annotations

import cv2
import numpy as np

from .contain_split import CONTAIN, MIN_PIECE, _contained
from .disjoint_add import mask_iou
from .rle import decode_rle, encode_rle


def _independent(hits: list[np.ndarray]) -> list[np.ndarray]:
    kept: list[np.ndarray] = []
    for hit in hits:
        if any(mask_iou(hit, previous) > CONTAIN for previous in kept):
            continue
        kept.append(hit)
    return kept


def _assign(binary: np.ndarray, hits: list[np.ndarray]) -> list[np.ndarray]:
    distances = []
    for hit in hits:
        outside = np.where(hit > 0, 0, 1).astype(np.uint8)
        distances.append(cv2.distanceTransform(outside, cv2.DIST_L2, 3))
    owner = np.argmin(np.stack(distances, axis=0), axis=0)
    pieces = []
    on = binary > 0
    for index in range(len(hits)):
        piece = (on & (owner == index)).astype(np.uint8)
        if int(piece.sum()) >= MIN_PIECE:
            pieces.append(piece)
    if len(pieces) < 2:
        return [binary]
    return pieces


def split_large(
    parent: np.ndarray,
    others: list[np.ndarray],
    cap: float,
) -> list[np.ndarray]:
    binary = (np.asarray(parent) > 0).astype(np.uint8)
    if float(binary.sum()) <= cap:
        return [binary]
    prepared = [(np.asarray(other) > 0).astype(np.uint8) for other in others]
    hits = _independent(_contained(binary, prepared))
    if len(hits) < 2:
        return [binary]
    return _assign(binary, hits)


def split_large_rles(
    parent_rles: list[str],
    other_rles: list[str],
    cap: float,
) -> tuple[list[str], int]:
    others = [decode_rle(rle) for rle in other_rles]
    out: list[str] = []
    splits = 0
    for rle in parent_rles:
        pieces = split_large(decode_rle(rle), others, cap)
        if len(pieces) > 1:
            splits += 1
        out.extend(encode_rle(piece) for piece in pieces)
    return out, splits
