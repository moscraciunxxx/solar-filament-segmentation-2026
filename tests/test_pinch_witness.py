"""A neck splits only when each piece matches its own witness mask."""

from __future__ import annotations

import unittest

import numpy as np

from filament.paths import IMAGE_SIZE
from filament.pinch_witness import WITNESS_IOU, pinch_if_witnessed


def _box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class PinchWitnessTest(unittest.TestCase):
    def test_both_witnesses_split_one_witness_stays_whole(self) -> None:
        joined = _box(100, 140, 100, 140)
        joined[119:121, 140:160] = 1
        joined[100:140, 160:200] = 1
        left = _box(100, 140, 100, 140)
        right = _box(100, 140, 160, 200)
        split = pinch_if_witnessed(joined, [left, right])
        self.assertEqual(WITNESS_IOU, 0.5)
        self.assertEqual(len(split), 2)
        self.assertNotEqual(int(split[0][120, 120]), int(split[0][120, 180]))
        whole = pinch_if_witnessed(joined, [left])
        self.assertEqual(len(whole), 1)
        self.assertEqual(int(whole[0][120, 120]), 1)
        self.assertEqual(int(whole[0][120, 180]), 1)


if __name__ == "__main__":
    unittest.main()
