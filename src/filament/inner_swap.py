"""Replace a mask with a strictly smaller mask that sits entirely inside it.

Each inner mask replaces at most one base, the smallest base that fully
covers it. An inner mask that sticks out of the base is left unused.
"""

from __future__ import annotations

import numpy as np


def inner_swap_masks(
    bases: list[np.ndarray],
    inners: list[np.ndarray],
) -> list[np.ndarray]:
    if not bases:
        return []
    candidates: list[tuple[int, int, int]] = []
    for base_index, base in enumerate(bases):
        cover = base > 0
        area = int(cover.sum())
        if area == 0:
            continue
        for inner_index, inner in enumerate(inners):
            core = inner > 0
            inner_area = int(core.sum())
            if inner_area == 0 or inner_area >= area:
                continue
            if bool(np.all(cover[core])):
                candidates.append((area, base_index, inner_index))
    candidates.sort()
    used_bases: set[int] = set()
    used_inners: set[int] = set()
    replace: dict[int, int] = {}
    for _area, base_index, inner_index in candidates:
        if base_index in used_bases or inner_index in used_inners:
            continue
        used_bases.add(base_index)
        used_inners.add(inner_index)
        replace[base_index] = inner_index
    swapped: list[np.ndarray] = []
    for base_index, base in enumerate(bases):
        if base_index in replace:
            swapped.append((inners[replace[base_index]] > 0).astype(np.uint8))
        else:
            swapped.append((base > 0).astype(np.uint8))
    return swapped
