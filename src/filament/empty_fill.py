"""Fill an image only when the 2048 model predicted nothing.

The fill is a pixel both the year-shift model and the all-annotator model
mark. Images that already have a 2048 mask are returned unchanged.
"""

from __future__ import annotations

from .vote import components, vote_map


def fill_empty(
    parent_rles: list[str],
    shift_rles: list[str],
    all_rles: list[str],
) -> list[str]:
    if parent_rles:
        return list(parent_rles)
    if not shift_rles or not all_rles:
        return []
    return components(vote_map([shift_rles, all_rles]))
