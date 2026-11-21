"""Give shared pixels to the larger mask when neither mask contains the other.

A nested mask is left whole. Disjoint masks are left whole. The smaller
mask of a partial overlap loses only the pixels both masks mark.
"""

from __future__ import annotations

import numpy as np


def cede_partial_overlap(masks: list[np.ndarray]) -> list[np.ndarray]:
    covers = [(mask > 0) for mask in masks]
    areas = [int(cover.sum()) for cover in covers]
    ceded = [np.zeros(cover.shape, dtype=bool) for cover in covers]
    for left in range(len(covers)):
        if areas[left] == 0:
            continue
        for right in range(left + 1, len(covers)):
            if areas[right] == 0:
                continue
            shared = covers[left] & covers[right]
            if not np.any(shared):
                continue
            left_inside = bool(np.all(covers[right][covers[left]]))
            right_inside = bool(np.all(covers[left][covers[right]]))
            if left_inside or right_inside:
                continue
            if areas[left] < areas[right]:
                ceded[left] |= shared
            else:
                ceded[right] |= shared
    kept: list[np.ndarray] = []
    for cover, gone in zip(covers, ceded, strict=True):
        remain = cover & ~gone
        if int(remain.sum()) > 0:
            kept.append(remain.astype(np.uint8))
    return kept
