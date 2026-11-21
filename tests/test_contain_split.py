"""A parent that holds two separate instances splits. One instance stays whole."""

from __future__ import annotations

import unittest

import numpy as np

from filament.contain_split import CONTAIN, contain_split
from filament.paths import IMAGE_SIZE


def _box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class ContainSplitTest(unittest.TestCase):
    def test_two_contained_instances_split_and_one_does_not(self) -> None:
        parent = _box(100, 200, 100, 400)
        left = _box(120, 170, 120, 190)
        right = _box(120, 170, 300, 370)
        pieces = contain_split(parent, [left, right])
        self.assertEqual(CONTAIN, 0.5)
        self.assertEqual(len(pieces), 2)
        self.assertNotEqual(int(pieces[0][140, 150]), int(pieces[0][140, 330]))
        self.assertEqual(len(contain_split(parent, [left])), 1)
        mostly_out = _box(120, 170, 380, 520)
        self.assertEqual(len(contain_split(parent, [left, mostly_out])), 1)


if __name__ == "__main__":
    unittest.main()
