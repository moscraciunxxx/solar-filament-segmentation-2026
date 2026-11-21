"""Vote the 2048, year-shift, and all-annotator masks, then split merges.

A pixel is kept only when two of the three models mark it. A surviving mask
is split when it contains two all-annotator instances. The vote stays 2 of 3
and the containment cut stays 0.5.
"""

from __future__ import annotations

from .contain_split import contain_rles
from .vote import components, vote_map


def vote_then_contain(
    parent_rles: list[str],
    shift_rles: list[str],
    all_rles: list[str],
) -> list[str]:
    if not parent_rles and not shift_rles and not all_rles:
        return []
    voted = components(vote_map([parent_rles, shift_rles, all_rles]))
    pieces, _splits = contain_rles(voted, all_rles)
    return pieces
