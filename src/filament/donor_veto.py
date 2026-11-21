"""Drop a mask the donor never touches.

One shared pixel keeps the mask. Nothing is added. This is not the
triple-witness drop and it does not append the donor's own leftovers.
"""

from __future__ import annotations

import numpy as np

from .outside_band import _best_iou


def drop_missed_by_donor(
    masks: list[np.ndarray], donors: list[np.ndarray]
) -> tuple[list[np.ndarray], int]:
    """Return the masks that share a pixel with a donor, and how many were dropped."""
    kept: list[np.ndarray] = []
    dropped = 0
    for mask in masks:
        if donors and _best_iou(mask, donors) > 0.0:
            kept.append(mask.copy())
            continue
        dropped += 1
    return kept, dropped
