"""A long mark far from the parent is kept. A mark on the parent is not."""

from __future__ import annotations

import unittest

import numpy as np

from filament.empty_region import components_far_from
from filament.paths import IMAGE_SIZE


class EmptyRegionTest(unittest.TestCase):
    def test_far_bar_kept_and_overlapping_bar_dropped(self) -> None:
        binary = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        occupied = np.zeros_like(binary)
        occupied[100:140, 100:160] = 1
        binary[100:108, 100:180] = 1
        binary[800:808, 800:900] = 1
        kept = components_far_from(binary, occupied, exclude_px=16, min_length=40)
        self.assertEqual(len(kept), 1)
        self.assertGreater(int(kept[0][804, 850]), 0)
        self.assertEqual(int(kept[0][104, 150]), 0)


if __name__ == "__main__":
    unittest.main()
