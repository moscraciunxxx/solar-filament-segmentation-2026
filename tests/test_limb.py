"""The limb ring includes the inner edge and excludes the disk center."""

from __future__ import annotations

import unittest

import numpy as np

from filament.limb import limb_annulus
from filament.paths import IMAGE_SIZE


class LimbTest(unittest.TestCase):
    def test_ring_geometry(self) -> None:
        ring = limb_annulus((IMAGE_SIZE, IMAGE_SIZE))
        center = IMAGE_SIZE // 2
        self.assertEqual(int(ring[center, center]), 0)
        self.assertEqual(int(ring[center, center + 930]), 1)
        self.assertEqual(int(ring[center, center + 500]), 0)
        self.assertEqual(int(ring[0, 0]), 0)


if __name__ == "__main__":
    unittest.main()
