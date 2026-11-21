"""Only an oversized mask with two contained instances is split."""

from __future__ import annotations

import unittest

import numpy as np

from filament.large_split import split_large
from filament.paths import IMAGE_SIZE


def _box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class LargeSplitTest(unittest.TestCase):
    def test_small_stays_whole_and_large_with_two_instances_splits(self) -> None:
        small = _box(100, 130, 100, 140)
        left = _box(104, 124, 104, 124)
        right = _box(104, 124, 116, 136)
        self.assertEqual(len(split_large(small, [left, right], cap=5000)), 1)

        large = _box(100, 220, 100, 460)
        left = _box(120, 180, 130, 220)
        right = _box(120, 180, 320, 420)
        pieces = split_large(large, [left, right], cap=5000)
        self.assertEqual(len(pieces), 2)
        self.assertNotEqual(int(pieces[0][150, 160]), int(pieces[0][150, 360]))
        self.assertEqual(len(split_large(large, [left], cap=5000)), 1)


if __name__ == "__main__":
    unittest.main()
