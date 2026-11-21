"""Drop a small mask the covering partner does not hold.

The size cut is 256 pixels, fixed. A partner must already contain at least
half the mask. Larger masks stay even when no partner covers them. Nothing
is added.
"""

from __future__ import annotations

import numpy as np

from .partner_merge import CONTAIN, _inside_fraction, _span

SMALL_BARE = 256


def drop_small_uncovered(
    masks: list[np.ndarray], partners: list[np.ndarray]
) -> tuple[list[np.ndarray], int]:
    """Return the kept masks and how many small uncovered masks were dropped."""
    partner_on = [(partner > 0).astype(np.uint8) for partner in partners]
    partner_spans = [_span(partner) for partner in partner_on]
    kept: list[np.ndarray] = []
    dropped = 0
    for mask in masks:
        binary = (mask > 0).astype(np.uint8)
        span = _span(binary)
        area = 0 if span is None else span[4]
        covered = False
        if span is not None and area < SMALL_BARE:
            for index, partner_span in enumerate(partner_spans):
                if partner_span is None:
                    continue
                fraction = _inside_fraction(binary, span, partner_on[index], partner_span)
                if fraction >= CONTAIN:
                    covered = True
                    break
        if span is not None and area < SMALL_BARE and not covered:
            dropped += 1
            continue
        kept.append(binary)
    return kept, dropped
