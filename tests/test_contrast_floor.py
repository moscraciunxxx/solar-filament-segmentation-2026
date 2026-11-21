"""A faint mask is dropped and a dark mask is kept. The floor is supplied."""

from __future__ import annotations

import unittest

import numpy as np

from filament.contrast_floor import RING, drop_faint_rles, mask_contrast
from filament.rle import encode_rle


def _blob(image: np.ndarray, x: int, y: int, value: int) -> str:
    image[y : y + 12, x : x + 12] = value
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y : y + 12, x : x + 12] = 1
    return encode_rle(mask)


class ContrastFloorTest(unittest.TestCase):
    def test_ring_is_31_pixels(self) -> None:
        self.assertEqual(RING, 31)

    def test_dark_blob_clears_a_low_floor_and_faint_blob_does_not(self) -> None:
        image = np.full((128, 128), 100, dtype=np.uint8)
        dark = _blob(image, 16, 16, 70)
        faint = _blob(image, 80, 80, 96)
        dark_mask = np.zeros(image.shape, dtype=np.uint8)
        dark_mask[16:28, 16:28] = 1
        self.assertGreater(mask_contrast(dark_mask, image), 10)
        kept, dropped = drop_faint_rles([dark, faint], image, floor=10)
        self.assertEqual(dropped, 1)
        self.assertEqual(kept, [dark])


if __name__ == "__main__":
    unittest.main()
