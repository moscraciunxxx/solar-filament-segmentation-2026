"""A 1.15 scale keeps the center and increases the area."""

from __future__ import annotations

import unittest

import numpy as np

from filament.paths import IMAGE_SIZE
from filament.scale_mask import SCALE, scale_mask


class ScaleMaskTest(unittest.TestCase):
    def test_center_stays_and_area_grows(self) -> None:
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        mask[200:240, 300:360] = 1
        grown = scale_mask(mask, SCALE)
        self.assertEqual(int(grown[220, 330]), 1)
        self.assertGreater(int(grown.sum()), int(mask.sum()))


if __name__ == "__main__":
    unittest.main()
