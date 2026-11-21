"""Merge masks that sit inside one full-medoid instance.

A mask counts when at least half its pixels lie in that instance. Two or
more such masks become their own union. The partner outline is not copied.
A partner-only mask is not appended. The half cut stays fixed.
"""

from __future__ import annotations

import numpy as np

CONTAIN = 0.5


def _span(mask: np.ndarray) -> tuple[int, int, int, int, int] | None:
    ys = np.flatnonzero(mask.any(axis=1))
    if ys.size == 0:
        return None
    xs = np.flatnonzero(mask.any(axis=0))
    if xs.size == 0:
        return None
    return int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1, int(mask.sum())


def _inside_fraction(
    base: np.ndarray,
    base_span: tuple[int, int, int, int, int],
    partner: np.ndarray,
    partner_span: tuple[int, int, int, int, int],
) -> float:
    y0 = max(base_span[0], partner_span[0])
    y1 = min(base_span[1], partner_span[1])
    x0 = max(base_span[2], partner_span[2])
    x1 = min(base_span[3], partner_span[3])
    if y1 <= y0 or x1 <= x0 or base_span[4] <= 0:
        return 0.0
    inter = int(np.count_nonzero(base[y0:y1, x0:x1] & partner[y0:y1, x0:x1]))
    return inter / base_span[4]


def _assign_groups(
    base_on: list[np.ndarray],
    partner_on: list[np.ndarray],
) -> tuple[list[tuple[int, int, int, int, int] | None], dict[int, list[int]]]:
    base_spans = [_span(mask) for mask in base_on]
    partner_spans = [_span(mask) for mask in partner_on]
    groups: dict[int, list[int]] = {}
    for base_index, base_span in enumerate(base_spans):
        if base_span is None:
            continue
        best_partner = None
        best_fraction = 0.0
        for partner_index, partner_span in enumerate(partner_spans):
            if partner_span is None:
                continue
            fraction = _inside_fraction(
                base_on[base_index],
                base_span,
                partner_on[partner_index],
                partner_span,
            )
            if fraction >= CONTAIN and fraction > best_fraction:
                best_fraction = fraction
                best_partner = partner_index
        if best_partner is not None:
            groups.setdefault(best_partner, []).append(base_index)
    return base_spans, groups


def _shares_pixel(
    left: np.ndarray,
    left_span: tuple[int, int, int, int, int],
    right: np.ndarray,
    right_span: tuple[int, int, int, int, int],
) -> bool:
    y0 = max(left_span[0], right_span[0])
    y1 = min(left_span[1], right_span[1])
    x0 = max(left_span[2], right_span[2])
    x1 = min(left_span[3], right_span[3])
    if y1 <= y0 or x1 <= x0:
        return False
    return bool(np.any(left[y0:y1, x0:x1] & right[y0:y1, x0:x1]))


def _emit(
    base_on: list[np.ndarray],
    merge_of: dict[int, list[int]],
    consumed: set[int],
) -> tuple[list[np.ndarray], int]:
    merged: list[np.ndarray] = []
    n_merged = 0
    for base_index, base in enumerate(base_on):
        if base_index in merge_of:
            union = np.zeros(base.shape, dtype=np.uint8)
            for other_index in merge_of[base_index]:
                union = np.maximum(union, base_on[other_index].astype(np.uint8))
            merged.append(union)
            n_merged += 1
        elif base_index in consumed:
            continue
        else:
            merged.append(base.astype(np.uint8))
    return merged, n_merged


def merge_inside_partner(
    base_masks: list[np.ndarray],
    partner_masks: list[np.ndarray],
) -> tuple[list[np.ndarray], int]:
    """Return the bases, with co-contained masks replaced by their union."""
    base_on = [mask > 0 for mask in base_masks]
    partner_on = [mask > 0 for mask in partner_masks]
    _spans, groups = _assign_groups(base_on, partner_on)
    merge_of: dict[int, list[int]] = {}
    consumed: set[int] = set()
    for indexes in groups.values():
        if len(indexes) < 2:
            continue
        merge_of[min(indexes)] = indexes
        consumed.update(indexes)
    return _emit(base_on, merge_of, consumed)


def merge_disjoint_inside_partner(
    base_masks: list[np.ndarray],
    partner_masks: list[np.ndarray],
) -> tuple[list[np.ndarray], int]:
    """Join co-contained masks only when they share no pixel.

    Masks that overlap each other stay separate. The partner outline is not
    copied. The half cut stays fixed.
    """
    base_on = [mask > 0 for mask in base_masks]
    partner_on = [mask > 0 for mask in partner_masks]
    base_spans, groups = _assign_groups(base_on, partner_on)
    merge_of: dict[int, list[int]] = {}
    consumed: set[int] = set()
    for indexes in groups.values():
        singletons: list[int] = []
        for base_index in indexes:
            base_span = base_spans[base_index]
            if base_span is None:
                continue
            touches = False
            for other_index in indexes:
                if other_index == base_index:
                    continue
                other_span = base_spans[other_index]
                if other_span is None:
                    continue
                if _shares_pixel(base_on[base_index], base_span, base_on[other_index], other_span):
                    touches = True
                    break
            if not touches:
                singletons.append(base_index)
        if len(singletons) < 2:
            continue
        merge_of[min(singletons)] = singletons
        consumed.update(singletons)
    return _emit(base_on, merge_of, consumed)
