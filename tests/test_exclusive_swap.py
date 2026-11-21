"""The second partner may replace only a mask the first partner left untouched."""

from __future__ import annotations

import unittest

import numpy as np

from filament.exclusive_swap import exclusive_contained_swap


class ExclusiveSwapTest(unittest.TestCase):
    def test_empty_bases_stay_empty(self) -> None:
        partner = np.ones((8, 8), dtype=np.uint8)
        self.assertEqual(exclusive_contained_swap([], [partner], [partner]), [])

    def test_primary_wins_when_both_contain_the_base(self) -> None:
        base = np.zeros((24, 24), dtype=np.uint8)
        base[8:16, 8:16] = 1
        primary = np.zeros((24, 24), dtype=np.uint8)
        primary[6:18, 6:18] = 1
        secondary = np.zeros((24, 24), dtype=np.uint8)
        secondary[2:22, 2:22] = 1
        swapped = exclusive_contained_swap([base], [primary], [secondary])[0]
        self.assertTrue(np.array_equal(swapped, primary))

    def test_secondary_fills_a_base_the_primary_does_not_cover(self) -> None:
        covered = np.zeros((24, 40), dtype=np.uint8)
        covered[4:10, 4:10] = 1
        open_base = np.zeros((24, 40), dtype=np.uint8)
        open_base[4:12, 24:34] = 1
        primary = np.zeros((24, 40), dtype=np.uint8)
        primary[2:14, 2:14] = 1
        secondary = np.zeros((24, 40), dtype=np.uint8)
        secondary[2:16, 20:38] = 1
        swapped = exclusive_contained_swap(
            [covered, open_base], [primary], [secondary]
        )
        self.assertTrue(np.array_equal(swapped[0], primary))
        self.assertTrue(np.array_equal(swapped[1], secondary))

    def test_a_partner_that_misses_pixels_is_ignored(self) -> None:
        base = np.zeros((16, 16), dtype=np.uint8)
        base[2:8, 2:8] = 1
        partial = np.zeros((16, 16), dtype=np.uint8)
        partial[2:6, 2:10] = 1
        swapped = exclusive_contained_swap([base], [partial], [partial])[0]
        self.assertTrue(np.array_equal(swapped, base))


if __name__ == "__main__":
    unittest.main()
