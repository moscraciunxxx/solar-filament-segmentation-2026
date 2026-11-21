"""Replace a mask with a full-medoid outline only when the two already agree.

The cut is IoU strictly above one half. A partner is used once, for the base
it overlaps most. A partner that does not clear the cut is not appended.
Nothing in an unmatched base is removed.
"""

from __future__ import annotations

import numpy as np

IOU_CUT = 0.5


def _span(mask: np.ndarray) -> tuple[int, int, int, int, int] | None:
    ys = np.flatnonzero(mask.any(axis=1))
    if ys.size == 0:
        return None
    xs = np.flatnonzero(mask.any(axis=0))
    if xs.size == 0:
        return None
    return int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1, int(mask.sum())


def _iou(
    base: np.ndarray,
    base_span: tuple[int, int, int, int, int],
    partner: np.ndarray,
    partner_span: tuple[int, int, int, int, int],
) -> float:
    y0 = max(base_span[0], partner_span[0])
    y1 = min(base_span[1], partner_span[1])
    x0 = max(base_span[2], partner_span[2])
    x1 = min(base_span[3], partner_span[3])
    if y1 <= y0 or x1 <= x0:
        return 0.0
    room = min((y1 - y0) * (x1 - x0), base_span[4], partner_span[4])
    if room / (base_span[4] + partner_span[4] - room) <= IOU_CUT:
        return 0.0
    inter = int(np.count_nonzero(base[y0:y1, x0:x1] & partner[y0:y1, x0:x1]))
    union = base_span[4] + partner_span[4] - inter
    if union <= 0:
        return 0.0
    return inter / union


def swap_agreed_outline(
    base_masks: list[np.ndarray],
    partner_masks: list[np.ndarray],
) -> tuple[list[np.ndarray], int]:
    """Return base masks, with a partner outline where IoU is above one half."""
    base_spans = [_span(mask > 0) for mask in base_masks]
    partner_spans = [_span(mask > 0) for mask in partner_masks]
    pairs: list[tuple[float, int, int]] = []
    for base_index, base_span in enumerate(base_spans):
        if base_span is None:
            continue
        base = base_masks[base_index] > 0
        for partner_index, partner_span in enumerate(partner_spans):
            if partner_span is None:
                continue
            overlap = _iou(base, base_span, partner_masks[partner_index] > 0, partner_span)
            if overlap > IOU_CUT:
                pairs.append((overlap, base_index, partner_index))
    pairs.sort(reverse=True)
    used_bases: set[int] = set()
    used_partners: set[int] = set()
    replace: dict[int, int] = {}
    for _overlap, base_index, partner_index in pairs:
        if base_index in used_bases or partner_index in used_partners:
            continue
        used_bases.add(base_index)
        used_partners.add(partner_index)
        replace[base_index] = partner_index
    swapped: list[np.ndarray] = []
    for base_index, base in enumerate(base_masks):
        if base_index in replace:
            swapped.append((partner_masks[replace[base_index]] > 0).astype(np.uint8))
        else:
            swapped.append((base > 0).astype(np.uint8))
    return swapped, len(replace)
