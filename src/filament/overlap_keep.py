"""Drop the smaller mask whenever two masks share a pixel.

The host scores a file that already has no overlapping predictions. A
containment swap can enlarge two masks until they touch. This keeps the
larger mask of every touching pair and leaves disjoint masks unchanged.
"""

from __future__ import annotations

import numpy as np


def keep_larger_disjoint(masks: list[np.ndarray]) -> list[np.ndarray]:
    kept = [(mask > 0).astype(np.uint8) for mask in masks if int((mask > 0).sum()) > 0]
    while True:
        drop = None
        for left in range(len(kept)):
            for right in range(left + 1, len(kept)):
                if not np.any(kept[left] & kept[right]):
                    continue
                left_area = int(kept[left].sum())
                right_area = int(kept[right].sum())
                if left_area < right_area:
                    drop = left
                elif right_area < left_area:
                    drop = right
                else:
                    drop = right
                break
            if drop is not None:
                break
        if drop is None:
            return kept
        del kept[drop]
