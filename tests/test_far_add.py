"""A far donor is kept; a neighbor, an overlap, and a blank image are not."""

from __future__ import annotations

import unittest

import numpy as np

from filament.far_add import FAR_PX, far_add
from filament.rle import encode_rle


def _blob(x: int, y: int, width: int = 4) -> str:
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[y : y + width, x : x + width] = 1
    return encode_rle(mask)


class FarAddTest(unittest.TestCase):
    def test_fixed_distance_is_64(self) -> None:
        self.assertEqual(FAR_PX, 64)

    def test_far_donor_is_appended_and_near_or_overlap_is_not(self) -> None:
        parent = _blob(100, 100)
        far = _blob(100, 400)
        near = _blob(100, 120)
        overlap = _blob(101, 101)
        kept, n_added = far_add([parent], [near, overlap, far])
        self.assertEqual(n_added, 1)
        self.assertEqual(kept, [parent, far])

    def test_blank_parent_adds_nothing(self) -> None:
        kept, n_added = far_add([], [_blob(100, 400)])
        self.assertEqual(kept, [])
        self.assertEqual(n_added, 0)


if __name__ == "__main__":
    unittest.main()
