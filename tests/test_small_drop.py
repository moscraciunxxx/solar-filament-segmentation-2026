"""Small masks are dropped and large masks are kept."""

from __future__ import annotations

import unittest

import numpy as np

from filament.paths import IMAGE_SIZE
from filament.rle import encode_rle
from filament.small_drop import MIN_AREA, drop_small_rles


def _rect(h: int, w: int) -> str:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[20 : 20 + h, 20 : 20 + w] = 1
    return encode_rle(mask)


class SmallDropTest(unittest.TestCase):
    def test_below_cutoff_dropped_and_above_kept(self) -> None:
        small = _rect(4, 4)
        large = _rect(20, 20)
        self.assertLess(16, MIN_AREA)
        self.assertGreater(400, MIN_AREA)
        self.assertEqual(drop_small_rles([small, large]), [large])


if __name__ == "__main__":
    unittest.main()
