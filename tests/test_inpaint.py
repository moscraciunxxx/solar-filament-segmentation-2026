"""Parent pixels are replaced by the outside median. Outside pixels stay."""

from __future__ import annotations

import unittest

import numpy as np

from filament.inpaint import inpaint_parent
from filament.paths import IMAGE_SIZE
from filament.rle import encode_rle


class InpaintTest(unittest.TestCase):
    def test_mask_becomes_median_and_corner_stays(self) -> None:
        image = np.full((IMAGE_SIZE, IMAGE_SIZE), 180, dtype=np.uint8)
        image[40:80, 40:90] = 20
        rle = encode_rle((image < 50).astype(np.uint8))
        painted = inpaint_parent(image, [rle])
        self.assertEqual(int(painted[50, 50]), 180)
        self.assertEqual(int(painted[0, 0]), 180)
        self.assertEqual(int(image[50, 50]), 20)


if __name__ == "__main__":
    unittest.main()
