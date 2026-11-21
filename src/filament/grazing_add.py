"""Append a donor mask only when it grazes the parent set.

The band is fixed: best IoU strictly above 0.05 and strictly below 0.35.
A pixel-disjoint donor is not added. A near-copy is not added. Every parent
mask stays.
"""

from __future__ import annotations

import numpy as np

from .disjoint_add import mask_iou

LO = 0.05
HI = 0.35


def add_grazing_donor(
    parent_masks: list[np.ndarray], donor_masks: list[np.ndarray]
) -> tuple[list[np.ndarray], int]:
    """Return the parent masks plus donors whose best overlap sits in the band."""
    kept = [mask.copy() for mask in parent_masks]
    added = 0
    for donor in donor_masks:
        if not parent_masks:
            continue
        best = max(mask_iou(donor, parent) for parent in parent_masks)
        if LO < best < HI:
            kept.append(donor.copy())
            added += 1
    return kept, added
