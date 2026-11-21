"""Hole fill closes an interior cavity and leaves a solid mask alone."""

from __future__ import annotations

import unittest

import numpy as np

from filament.holes import fill_holes
from filament.paths import IMAGE_SIZE


def _blank() -> np.ndarray:
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)


class HoleFillTest(unittest.TestCase):
    def test_ring_is_filled_and_solid_is_unchanged(self) -> None:
        ring = _blank()
        ring[40:80, 40:80] = 1
        ring[50:70, 50:70] = 0
        filled = fill_holes(ring)
        self.assertEqual(int(filled[55, 55]), 1)
        self.assertEqual(int(filled[10, 10]), 0)
        self.assertGreater(int(filled.sum()), int(ring.sum()))

        solid = _blank()
        solid[100:130, 100:140] = 1
        self.assertEqual(int(fill_holes(solid).sum()), int(solid.sum()))


if __name__ == "__main__":
    unittest.main()
