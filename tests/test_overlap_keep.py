"""Overlapping masks keep the larger one. Disjoint masks stay."""

from __future__ import annotations

import unittest

import numpy as np

from filament.overlap_keep import keep_larger_disjoint


class OverlapKeepTest(unittest.TestCase):
    def test_empty_stays_empty(self) -> None:
        self.assertEqual(keep_larger_disjoint([]), [])

    def test_disjoint_masks_both_stay(self) -> None:
        left = np.zeros((20, 20), dtype=np.uint8)
        right = np.zeros((20, 20), dtype=np.uint8)
        left[1:5, 1:5] = 1
        right[12:18, 12:18] = 1
        kept = keep_larger_disjoint([left, right])
        self.assertEqual(len(kept), 2)
        self.assertTrue(np.array_equal(kept[0], left))
        self.assertTrue(np.array_equal(kept[1], right))

    def test_overlap_drops_the_smaller_mask(self) -> None:
        small = np.zeros((24, 24), dtype=np.uint8)
        large = np.zeros((24, 24), dtype=np.uint8)
        small[0:8, 0:8] = 1
        large[4:20, 4:20] = 1
        kept = keep_larger_disjoint([small, large])
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], large))

    def test_one_large_mask_clears_two_overlaps(self) -> None:
        large = np.zeros((30, 30), dtype=np.uint8)
        left = np.zeros((30, 30), dtype=np.uint8)
        right = np.zeros((30, 30), dtype=np.uint8)
        large[5:25, 5:25] = 1
        left[0:10, 0:10] = 1
        right[20:28, 20:28] = 1
        kept = keep_larger_disjoint([left, large, right])
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], large))


if __name__ == "__main__":
    unittest.main()
