"""The foreground marker is still present after the basin claim."""

from __future__ import annotations

import unittest

import numpy as np

from filament.marker_watershed import marker_watershed
from filament.paths import IMAGE_SIZE


class MarkerWatershedTest(unittest.TestCase):
    def test_seed_survives_on_a_dark_filament(self) -> None:
        image = np.full((IMAGE_SIZE, IMAGE_SIZE), 200, dtype=np.uint8)
        image[90:150, 80:220] = 35
        mask = np.zeros_like(image)
        mask[110:130, 120:170] = 1
        refined = marker_watershed(image, mask)
        self.assertEqual(int(refined[120, 140]), 1)
        self.assertGreaterEqual(int(refined.sum()), int(mask.sum()))


if __name__ == "__main__":
    unittest.main()
