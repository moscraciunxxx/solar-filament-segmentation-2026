"""Keep a 2-of-3 vote component only when it touches a 2048 mask.

The voters are the 2048 model, the year-shift model, and the all-annotator
model. The vote stays 2 of 3. A component that shares no pixel with the
2048 model is dropped. Outlines of kept components are not rewritten.
"""

from __future__ import annotations

import numpy as np

from .rle import decode_rle
from .vote import components, vote_map


def _touches(mask: np.ndarray, parents: list[np.ndarray]) -> bool:
    if not parents:
        return False
    flat = mask.ravel().astype(np.float32)
    stacked = np.stack([parent.ravel() for parent in parents]).astype(np.float32)
    intersection = stacked @ flat
    return bool(np.any(intersection > 0))


def vote_supported(
    parent_rles: list[str],
    shift_rles: list[str],
    all_rles: list[str],
) -> list[str]:
    if not parent_rles and not shift_rles and not all_rles:
        return []
    voted = components(vote_map([parent_rles, shift_rles, all_rles]))
    if not parent_rles:
        return []
    parents = [decode_rle(rle) for rle in parent_rles]
    return [rle for rle in voted if _touches(decode_rle(rle), parents)]
