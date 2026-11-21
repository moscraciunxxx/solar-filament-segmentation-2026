"""A 7-pixel close bridges a 3-pixel gap and leaves a far gap open."""

from __future__ import annotations

import unittest

import numpy as np

from filament.close_mask import KERNEL, close_mask
from filament.paths import IMAGE_SIZE


class CloseMaskTest(unittest.TestCase):
    def test_near_gap_closes_and_far_gap_stays(self) -> None:
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        mask[100:110, 40:70] = 1
        mask[100:110, 73:110] = 1
        mask[100:110, 200:240] = 1
        closed = close_mask(mask, KERNEL)
        self.assertEqual(int(closed[104, 71]), 1)
        self.assertEqual(int(closed[104, 160]), 0)


if __name__ == "__main__":
    unittest.main()
