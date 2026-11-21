"""Let a second partner replace only masks the first partner left untouched.

The first partner is applied with the locked containment rule. A mask it
replaces is frozen. The second partner then uses the same rule on whatever
remains. Neither partner may miss a pixel or be the same size or smaller.
"""

from __future__ import annotations

import numpy as np

from filament.contained_swap import contained_swap_masks


def exclusive_contained_swap(
    bases: list[np.ndarray],
    primary: list[np.ndarray],
    secondary: list[np.ndarray],
) -> list[np.ndarray]:
    if not bases:
        return []
    first = contained_swap_masks(bases, primary)
    kept_index: list[int] = []
    kept: list[np.ndarray] = []
    for index, (before, after) in enumerate(zip(bases, first, strict=True)):
        if np.array_equal(before > 0, after > 0):
            kept_index.append(index)
            kept.append(before)
    second = contained_swap_masks(kept, secondary)
    swapped = list(first)
    for index, mask in zip(kept_index, second, strict=True):
        swapped[index] = mask
    return swapped
