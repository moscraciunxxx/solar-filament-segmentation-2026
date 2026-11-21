"""Fill a short gap inside the partner that already joins two pieces.

The radius is the locked 8-pixel tip distance. A new pixel is kept only when
it lies in the partner and within that radius of two pieces of the same mask.
The rest of the partner is not copied.
"""

from __future__ import annotations

import cv2
import numpy as np

from .partner_merge import CONTAIN, _inside_fraction, _span

BRIDGE_PX = 8


def bridge_inside_partner(union: np.ndarray, partner: np.ndarray) -> np.ndarray:
    """Return the mask plus partner pixels both pieces can reach."""
    binary = (union > 0).astype(np.uint8)
    cover = partner > 0
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    if count < 3:
        return binary
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BRIDGE_PX * 2 + 1, BRIDGE_PX * 2 + 1),
    )
    result = binary.copy()
    for left in range(1, count):
        grown_left = cv2.dilate((labels == left).astype(np.uint8), kernel)
        for right in range(left + 1, count):
            grown_right = cv2.dilate((labels == right).astype(np.uint8), kernel)
            result[(grown_left > 0) & (grown_right > 0) & cover] = 1
    return result


def bridge_masks(
    masks: list[np.ndarray],
    partners: list[np.ndarray],
) -> tuple[list[np.ndarray], int]:
    """Bridge each mask toward the partner that already covers half of it."""
    partner_on = [mask > 0 for mask in partners]
    partner_spans = [_span(mask) for mask in partner_on]
    bridged: list[np.ndarray] = []
    n_bridged = 0
    for mask in masks:
        base = mask > 0
        base_span = _span(base)
        chosen = None
        best_fraction = 0.0
        if base_span is not None:
            for partner, partner_span in zip(partner_on, partner_spans, strict=True):
                if partner_span is None:
                    continue
                fraction = _inside_fraction(base, base_span, partner, partner_span)
                if fraction >= CONTAIN and fraction > best_fraction:
                    best_fraction = fraction
                    chosen = partner
        if chosen is None:
            bridged.append(base.astype(np.uint8))
            continue
        wider = bridge_inside_partner(base.astype(np.uint8), chosen.astype(np.uint8))
        if not np.array_equal(wider > 0, base):
            n_bridged += 1
        bridged.append(wider)
    return bridged, n_bridged
