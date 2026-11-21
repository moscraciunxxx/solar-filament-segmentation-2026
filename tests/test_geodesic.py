"""Growth follows a dark corridor and does not jump a bright gap or a far blob."""

from __future__ import annotations

import unittest

import numpy as np

from filament.geodesic import geodesic_extend
from filament.paths import IMAGE_SIZE


class GeodesicTest(unittest.TestCase):
    def test_follows_corridor_and_stops_at_a_gap(self) -> None:
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        dark = np.zeros_like(mask)
        mask[100:106, 40:50] = 1
        dark[100:106, 40:80] = 1
        dark[400:406, 400:460] = 1
        grown = geodesic_extend(mask, dark, steps=24)
        self.assertEqual(int(grown[102, 70]), 1)
        self.assertEqual(int(grown[402, 420]), 0)

        blocked = dark.copy()
        blocked[100:106, 55:60] = 0
        stopped = geodesic_extend(mask, blocked, steps=24)
        self.assertEqual(int(stopped[102, 70]), 0)


if __name__ == "__main__":
    unittest.main()
