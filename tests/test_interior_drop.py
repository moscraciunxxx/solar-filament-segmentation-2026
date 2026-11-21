"""A mask drops only when a sibling is strictly larger and covers every pixel."""

from __future__ import annotations

import unittest

import numpy as np

from filament.interior_drop import drop_strict_interior


class InteriorDropTest(unittest.TestCase):
    def test_empty_stays_empty(self) -> None:
        self.assertEqual(drop_strict_interior([]), [])

    def test_a_partial_overlap_keeps_both(self) -> None:
        left = np.zeros((20, 20), dtype=np.uint8)
        right = np.zeros((20, 20), dtype=np.uint8)
        left[2:12, 2:10] = 1
        right[6:16, 6:16] = 1
        kept = drop_strict_interior([left, right])
        self.assertEqual(len(kept), 2)
        self.assertTrue(np.array_equal(kept[0], left))
        self.assertTrue(np.array_equal(kept[1], right))

    def test_a_strict_interior_is_dropped(self) -> None:
        outer = np.zeros((24, 24), dtype=np.uint8)
        inner = np.zeros((24, 24), dtype=np.uint8)
        outer[2:22, 2:22] = 1
        inner[8:14, 8:14] = 1
        kept = drop_strict_interior([inner, outer])
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], outer))

    def test_equal_masks_both_stay(self) -> None:
        one = np.zeros((12, 12), dtype=np.uint8)
        one[2:8, 2:8] = 1
        kept = drop_strict_interior([one.copy(), one.copy()])
        self.assertEqual(len(kept), 2)

    def test_one_outer_drops_two_interiors(self) -> None:
        outer = np.zeros((30, 30), dtype=np.uint8)
        left = np.zeros((30, 30), dtype=np.uint8)
        right = np.zeros((30, 30), dtype=np.uint8)
        outer[2:28, 2:28] = 1
        left[4:10, 4:10] = 1
        right[18:24, 18:24] = 1
        kept = drop_strict_interior([left, outer, right])
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], outer))


if __name__ == "__main__":
    unittest.main()
