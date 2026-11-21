"""Slot1 erosion keeps a wide filament and drops a speck."""

from __future__ import annotations

import unittest

import numpy as np

from filament.yolo_export import erode_instance_polygon


def _rect(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


class ErodeGtKeepTest(unittest.TestCase):
    def test_wide_filament_survives_and_speck_vanishes(self) -> None:
        wide = erode_instance_polygon(_rect(100, 100, 180, 160))
        speck = erode_instance_polygon(_rect(400, 400, 402, 403))
        self.assertIsNotNone(wide)
        self.assertGreater(len(wide), 2)
        self.assertIsNone(speck)


if __name__ == "__main__":
    unittest.main()
