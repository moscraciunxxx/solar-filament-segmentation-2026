"""Replace a mask with a strictly larger mask that already contains it.

Each partner can replace only one base, the largest one it fully covers.
A partner that misses any base pixel is left unused. No overlap cut.
"""

from __future__ import annotations

import numpy as np


def contained_swap_masks(
    bases: list[np.ndarray],
    partners: list[np.ndarray],
) -> list[np.ndarray]:
    if not bases:
        return []
    candidates: list[tuple[int, int, int]] = []
    for base_index, base in enumerate(bases):
        core = base > 0
        area = int(core.sum())
        if area == 0:
            continue
        for partner_index, partner in enumerate(partners):
            cover = partner > 0
            if int(cover.sum()) <= area:
                continue
            if bool(np.all(cover[core])):
                candidates.append((area, base_index, partner_index))
    candidates.sort(reverse=True)
    used_bases: set[int] = set()
    used_partners: set[int] = set()
    replace: dict[int, int] = {}
    for _area, base_index, partner_index in candidates:
        if base_index in used_bases or partner_index in used_partners:
            continue
        used_bases.add(base_index)
        used_partners.add(partner_index)
        replace[base_index] = partner_index
    swapped: list[np.ndarray] = []
    for base_index, base in enumerate(bases):
        if base_index in replace:
            swapped.append((partners[replace[base_index]] > 0).astype(np.uint8))
        else:
            swapped.append((base > 0).astype(np.uint8))
    return swapped
