"""A partial overlap cedes the shared pixels. A nested mask stays whole."""

from __future__ import annotations

import unittest

import numpy as np

from filament.cede_overlap import cede_partial_overlap


class CedeOverlapTest(unittest.TestCase):
    def test_empty_stays_empty(self) -> None:
        self.assertEqual(cede_partial_overlap([]), [])

    def test_disjoint_masks_both_stay(self) -> None:
        left = np.zeros((24, 24), dtype=np.uint8)
        right = np.zeros((24, 24), dtype=np.uint8)
        left[1:6, 1:6] = 1
        right[14:22, 14:22] = 1
        kept = cede_partial_overlap([left, right])
        self.assertEqual(len(kept), 2)
        self.assertTrue(np.array_equal(kept[0], left))
        self.assertTrue(np.array_equal(kept[1], right))

    def test_a_strict_interior_is_left_whole(self) -> None:
        outer = np.zeros((24, 24), dtype=np.uint8)
        inner = np.zeros((24, 24), dtype=np.uint8)
        outer[2:22, 2:22] = 1
        inner[8:14, 8:14] = 1
        kept = cede_partial_overlap([inner, outer])
        self.assertEqual(len(kept), 2)
        self.assertTrue(np.array_equal(kept[0], inner))
        self.assertTrue(np.array_equal(kept[1], outer))

    def test_partial_overlap_drops_only_the_shared_pixels(self) -> None:
        large = np.zeros((30, 30), dtype=np.uint8)
        small = np.zeros((30, 30), dtype=np.uint8)
        large[2:20, 2:20] = 1
        small[16:28, 16:28] = 1
        kept = cede_partial_overlap([small, large])
        self.assertEqual(len(kept), 2)
        shared = (small > 0) & (large > 0)
        self.assertTrue(np.array_equal(kept[1], large))
        self.assertEqual(int(kept[0][shared].sum()), 0)
        self.assertGreater(int(kept[0].sum()), 0)
        self.assertTrue(bool(np.all(small[kept[0] > 0])))


if __name__ == "__main__":
    unittest.main()
