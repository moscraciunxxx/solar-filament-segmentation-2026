"""Keep a pixel only when at least two models mark it.

The vote is fixed at 2 of 3. Components smaller than 96 pixels are dropped.
This does not append one model's instance list onto another's.
"""

from __future__ import annotations

import cv2
import numpy as np

from .rle import decode_rle, encode_rle
from .small_drop import MIN_AREA

MIN_VOTES = 2


def vote_map(rle_sets: list[list[str]], votes: int = MIN_VOTES) -> np.ndarray:
    if not rle_sets:
        raise ValueError("need at least one model")
    canvas = None
    for rles in rle_sets:
        layer = None
        for rle in rles:
            mask = decode_rle(rle)
            layer = mask.astype(np.uint8) if layer is None else np.maximum(layer, mask)
        if layer is None:
            continue
        canvas = layer.astype(np.uint16) if canvas is None else canvas + layer
    if canvas is None:
        raise ValueError("every model was empty")
    return (canvas >= votes).astype(np.uint8)


def components(binary: np.ndarray, min_area: int = MIN_AREA) -> list[str]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    rles: list[str] = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < min_area:
            continue
        rles.append(encode_rle((labels == index).astype(np.uint8)))
    return rles
