"""Drop a mask only when a sibling strictly contains it.

A partial overlap stays. Equal areas stay. The surviving pixels are the
sibling that was already larger. No new threshold.
"""

from __future__ import annotations

import numpy as np


def drop_strict_interior(masks: list[np.ndarray]) -> list[np.ndarray]:
    covers = [(mask > 0) for mask in masks]
    areas = [int(cover.sum()) for cover in covers]
    drop: set[int] = set()
    for index, core in enumerate(covers):
        if areas[index] == 0:
            drop.add(index)
            continue
        for other_index, other in enumerate(covers):
            if other_index == index or areas[other_index] <= areas[index]:
                continue
            if bool(np.all(other[core])):
                drop.add(index)
                break
    return [
        cover.astype(np.uint8)
        for index, cover in enumerate(covers)
        if index not in drop and areas[index] > 0
    ]


def drop_nested_after_restore(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Drop a mask that is strictly inside a sibling after masks are restored.

    This is the same strict-interior rule. A partial overlap stays. Equal
    areas stay. No new threshold.
    """
    return drop_strict_interior(masks)
