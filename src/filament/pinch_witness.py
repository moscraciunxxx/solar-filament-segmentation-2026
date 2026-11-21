"""Split a neck only when each piece is its own object in another model.

The erosion stays 2 pixels. A piece counts only when some witness mask
overlaps it above 0.5, the contest true-positive cut. Otherwise the parent
mask is kept whole. This does not change the erosion.
"""

from __future__ import annotations

import numpy as np

from .pinch_split import pinch_split
from .rle import decode_rle, encode_rle

WITNESS_IOU = 0.5


def _best_iou(mask: np.ndarray, others: list[np.ndarray]) -> float:
    if not others:
        return 0.0
    flat = mask.ravel().astype(np.float32)
    stacked = np.stack([other.ravel() for other in others]).astype(np.float32)
    intersection = stacked @ flat
    union = stacked.sum(axis=1) + float(flat.sum()) - intersection
    overlaps = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return float(overlaps.max())


def pinch_if_witnessed(
    mask: np.ndarray,
    witnesses: list[np.ndarray],
    witness_iou: float = WITNESS_IOU,
) -> list[np.ndarray]:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    pieces = pinch_split(binary)
    if len(pieces) < 2:
        return [binary]
    if all(_best_iou(piece, witnesses) > witness_iou for piece in pieces):
        return pieces
    return [binary]


def pinch_witness_rles(parent_rles: list[str], witness_rles: list[str]) -> tuple[list[str], int]:
    witnesses = [decode_rle(rle) for rle in witness_rles]
    out: list[str] = []
    splits = 0
    for rle in parent_rles:
        mask = decode_rle(rle)
        pieces = pinch_if_witnessed(mask, witnesses)
        if len(pieces) > 1:
            splits += 1
        out.extend(encode_rle(piece) for piece in pieces)
    return out, splits
