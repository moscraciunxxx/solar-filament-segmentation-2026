"""A candidate is added only when it shares no pixel with the parent."""

from __future__ import annotations

import unittest

import numpy as np

from filament.disjoint_add import disjoint_add
from filament.rle import encode_rle


def _blob(x: int, y: int) -> str:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[y : y + 6, x : x + 6] = 1
    return encode_rle(mask)


class DisjointAddTest(unittest.TestCase):
    def test_overlap_is_dropped_and_far_mask_is_kept(self) -> None:
        parent = _blob(4, 4)
        overlap = _blob(6, 6)
        far = _blob(40, 40)
        kept, n_added = disjoint_add([parent], [overlap, far], image_size=64)
        self.assertEqual(n_added, 1)
        self.assertEqual(kept, [parent, far])

    def test_empty_candidate_list_keeps_the_parent(self) -> None:
        parent = _blob(4, 4)
        kept, n_added = disjoint_add([parent], [], image_size=64)
        self.assertEqual(kept, [parent])
        self.assertEqual(n_added, 0)


if __name__ == "__main__":
    unittest.main()
