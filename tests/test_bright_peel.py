"""A bright outer pixel is removed. A dark outer pixel stays."""

from __future__ import annotations

import unittest

import numpy as np

from filament.bright_peel import MARGIN, bright_peel
from filament.paths import IMAGE_SIZE


class BrightPeelTest(unittest.TestCase):
    def test_bright_edge_drops_and_dark_edge_stays(self) -> None:
        image = np.full((IMAGE_SIZE, IMAGE_SIZE), 140, dtype=np.uint8)
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        mask[500:520, 400:480] = 1
        image[mask > 0] = 90
        image[500:520, 479] = 160
        peeled = bright_peel(image, mask)
        self.assertEqual(MARGIN, 12)
        self.assertEqual(int(peeled[510, 450]), 1)
        self.assertEqual(int(peeled[510, 400]), 1)
        self.assertEqual(int(peeled[510, 479]), 0)


if __name__ == "__main__":
    unittest.main()
