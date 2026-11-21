"""A two-pixel neck splits. A solid bar stays one mask."""

from __future__ import annotations

import unittest

import numpy as np

from filament.paths import IMAGE_SIZE
from filament.pinch_split import ERODE, pinch_split


class PinchSplitTest(unittest.TestCase):
    def test_neck_splits_and_solid_bar_does_not(self) -> None:
        joined = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        joined[100:140, 100:140] = 1
        joined[119:121, 140:160] = 1
        joined[100:140, 160:200] = 1
        pieces = pinch_split(joined)
        self.assertEqual(ERODE, 2)
        self.assertEqual(len(pieces), 2)
        self.assertEqual(int(pieces[0][120, 120]) + int(pieces[1][120, 120]), 1)
        self.assertEqual(int(pieces[0][120, 180]) + int(pieces[1][120, 180]), 1)
        self.assertNotEqual(int(pieces[0][120, 120]), int(pieces[0][120, 180]))

        bar = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        bar[300:340, 300:500] = 1
        self.assertEqual(len(pinch_split(bar)), 1)
        self.assertTrue(np.array_equal(pinch_split(bar)[0], bar))


if __name__ == "__main__":
    unittest.main()
