"""GrabCut keeps the seed pixels of a dark filament on a bright disk."""

from __future__ import annotations

import unittest

import numpy as np

from filament.grab import refine_mask
from filament.paths import IMAGE_SIZE


class GrabTest(unittest.TestCase):
    def test_seed_survives(self) -> None:
        image = np.full((IMAGE_SIZE, IMAGE_SIZE), 190, dtype=np.uint8)
        image[80:140, 80:200] = 40
        mask = np.zeros_like(image)
        mask[100:120, 110:160] = 1
        refined = refine_mask(image, mask)
        self.assertEqual(int(refined[110, 130]), 1)
        self.assertGreaterEqual(int(refined.sum()), int(mask.sum()))


if __name__ == "__main__":
    unittest.main()
