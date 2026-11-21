"""Remove pixels that sit outside the partner covering a mask.

A partner is used only when at least half the mask already lies inside it.
The partner outline is not copied. A mask below that half cut is unchanged.
"""

from __future__ import annotations

import numpy as np

from .partner_merge import CONTAIN, _inside_fraction, _span


def clip_outside_partner(
    masks: list[np.ndarray], partners: list[np.ndarray]
) -> tuple[list[np.ndarray], int]:
    """Return masks with outside pixels removed, and how many masks changed."""
    partner_on = [(partner > 0).astype(np.uint8) for partner in partners]
    partner_spans = [_span(partner) for partner in partner_on]
    kept: list[np.ndarray] = []
    changed = 0
    for mask in masks:
        binary = (mask > 0).astype(np.uint8)
        span = _span(binary)
        best = None
        best_fraction = 0.0
        if span is not None:
            for index, partner_span in enumerate(partner_spans):
                if partner_span is None:
                    continue
                fraction = _inside_fraction(binary, span, partner_on[index], partner_span)
                if fraction >= CONTAIN and fraction > best_fraction:
                    best_fraction = fraction
                    best = index
        if best is None:
            kept.append(binary)
            continue
        clipped = np.logical_and(binary, partner_on[best]).astype(np.uint8)
        if not np.array_equal(clipped, binary):
            changed += 1
        kept.append(clipped)
    return kept, changed
