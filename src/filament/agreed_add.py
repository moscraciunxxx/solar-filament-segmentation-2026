"""Add a component only where two donor models agree and the base is empty.

Votes stay at 2. Components under 96 pixels are dropped. A candidate that
shares a pixel with the base is left out. Nothing in the base is removed.
"""

from __future__ import annotations

import cv2
import numpy as np

from .contrast_floor import mask_contrast
from .disjoint_add import mask_iou
from .small_drop import MIN_AREA
from .vote import MIN_VOTES


def add_agreed_masks(
    base_masks: list[np.ndarray],
    donor_groups: list[list[np.ndarray]],
    votes: int = MIN_VOTES,
    min_area: int = MIN_AREA,
) -> tuple[list[np.ndarray], int]:
    """Return the base masks plus agreed components that miss every one of them."""
    kept = [mask.copy() for mask in base_masks]
    shape = kept[0].shape if kept else None
    if shape is None:
        for group in donor_groups:
            if group:
                shape = group[0].shape
                break
    if shape is None:
        return kept, 0
    canvas = np.zeros(shape, dtype=np.uint16)
    saw_donor = False
    for group in donor_groups:
        if not group:
            continue
        layer = np.zeros(shape, dtype=np.uint8)
        for mask in group:
            layer = np.maximum(layer, (mask > 0).astype(np.uint8))
        if int(layer.sum()) == 0:
            continue
        saw_donor = True
        canvas += layer
    if not saw_donor:
        return kept, 0
    binary = (canvas >= votes).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    added = 0
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < min_area:
            continue
        component = (labels == index).astype(np.uint8)
        if any(mask_iou(component, other) != 0.0 for other in kept):
            continue
        kept.append(component)
        added += 1
    return kept, added


def add_unanimous_masks(
    base_masks: list[np.ndarray],
    donor_groups: list[list[np.ndarray]],
    min_area: int = MIN_AREA,
) -> tuple[list[np.ndarray], int]:
    """Add a component only where every donor marks the pixel and the base misses it.

    The vote count is the number of donors. One missing donor adds nothing.
    Components under 96 pixels are dropped. Nothing in the base is removed.
    """
    if not donor_groups:
        return [mask.copy() for mask in base_masks], 0
    return add_agreed_masks(
        base_masks,
        donor_groups,
        votes=len(donor_groups),
        min_area=min_area,
    )


def keep_added_if_touched(
    base_masks: list[np.ndarray],
    added_masks: list[np.ndarray],
    witness_masks: list[np.ndarray],
) -> tuple[list[np.ndarray], int]:
    """Keep every base mask. Keep an added mask only when the witness shares a pixel.

    Nothing from the witness is appended. The zero-pixel test is exact.
    """
    kept = [mask.copy() for mask in base_masks]
    kept_added = 0
    for mask in added_masks:
        if any(mask_iou(mask, other) > 0.0 for other in witness_masks):
            kept.append(mask.copy())
            kept_added += 1
    return kept, kept_added


def drop_faint_adds(
    base_masks: list[np.ndarray],
    added_masks: list[np.ndarray],
    image: np.ndarray,
    floor: float,
) -> tuple[list[np.ndarray], int]:
    """Keep every base mask. Drop an added mask only when it is fainter than the floor.

    A mask whose ring cannot be measured stays. The floor is not chosen here.
    """
    kept = [mask.copy() for mask in base_masks]
    kept_added = 0
    for mask in added_masks:
        contrast = mask_contrast(mask, image)
        if contrast is not None and contrast < floor:
            continue
        kept.append(mask.copy())
        kept_added += 1
    return kept, kept_added


def extend_core_to_pair_fringe(
    core: np.ndarray,
    donor_masks: list[np.ndarray],
    base_masks: list[np.ndarray],
) -> np.ndarray:
    """Grow one core into the two-donor component that contains it.

    Pixels already in a base mask stay out. A separate two-donor blob that
    does not contain the core is not absorbed. The vote stays at two.
    """
    core_on = core > 0
    canvas = np.zeros(core.shape, dtype=np.uint16)
    for mask in donor_masks:
        canvas += (mask > 0).astype(np.uint8)
    binary = (canvas >= MIN_VOTES).astype(np.uint8)
    _count, labels = cv2.connectedComponents(binary, connectivity=8)
    chosen = np.zeros(core.shape, dtype=np.uint8)
    if int(core_on.sum()) > 0:
        hits = {int(value) for value in np.unique(labels[core_on]) if int(value) != 0}
        if hits:
            chosen = np.isin(labels, list(hits)).astype(np.uint8)
        else:
            chosen = core_on.astype(np.uint8)
    if base_masks:
        blocked = np.zeros(core.shape, dtype=bool)
        for mask in base_masks:
            blocked |= mask > 0
        chosen[blocked] = 0
        surviving = np.logical_and(core_on, chosen > 0)
        if int(surviving.sum()) == 0:
            return np.zeros(core.shape, dtype=np.uint8)
        _count2, labels2 = cv2.connectedComponents(chosen, connectivity=8)
        keep = {int(value) for value in np.unique(labels2[surviving]) if int(value) != 0}
        chosen = np.isin(labels2, list(keep)).astype(np.uint8)
    return chosen
