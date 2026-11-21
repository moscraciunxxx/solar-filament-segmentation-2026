"""A dark line has a stronger ridge response than a flat bright patch."""

from __future__ import annotations

import unittest

import numpy as np

from filament.paths import IMAGE_SIZE
from filament.ridge import dark_ridge


class RidgeTest(unittest.TestCase):
    def test_dark_line_outranks_flat_disk(self) -> None:
        image = np.full((IMAGE_SIZE, IMAGE_SIZE), 180, dtype=np.uint8)
        image[400:406, 200:700] = 30
        response = dark_ridge(image, sigma=3.0)
        self.assertGreater(float(response[402, 400]), float(response[20, 20]))


if __name__ == "__main__":
    unittest.main()
