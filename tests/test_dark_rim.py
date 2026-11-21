"""A dark pixel within 3px is added. A bright neighbor and a far dark pixel are not."""

from __future__ import annotations

import unittest

import numpy as np

from filament.dark_rim import MARGIN, RIM, dark_rim
from filament.paths import IMAGE_SIZE


class DarkRimTest(unittest.TestCase):
    def test_dark_neighbor_joins_and_bright_or_far_pixel_does_not(self) -> None:
        image = np.full((IMAGE_SIZE, IMAGE_SIZE), 140, dtype=np.uint8)
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        mask[500:520, 400:480] = 1
        image[500:520, 400:480] = 90
        image[500:520, 480:483] = 95
        image[510, 481] = 140
        image[500:520, 490:493] = 80
        grown = dark_rim(image, mask)
        self.assertEqual(RIM, 3)
        self.assertEqual(MARGIN, 8)
        self.assertEqual(int(grown[510, 450]), 1)
        self.assertEqual(int(grown[510, 482]), 1)
        self.assertEqual(int(grown[510, 481]), 0)
        self.assertEqual(int(grown[510, 491]), 0)


if __name__ == "__main__":
    unittest.main()
