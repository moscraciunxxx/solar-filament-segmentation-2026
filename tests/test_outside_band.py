"""A parent mask nobody else sees is dropped. A witnessed mask off the parent is added."""

from __future__ import annotations

import unittest

import numpy as np

from filament.outside_band import WITNESS_IOU, edit_outside_band
from filament.paths import IMAGE_SIZE


def _box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class OutsideBandTest(unittest.TestCase):
    def test_lone_parent_dropped_and_witnessed_outsider_added(self) -> None:
        lone = _box(20, 50, 20, 120)
        supported = _box(200, 240, 200, 320)
        outsider = _box(500, 540, 500, 680)
        junk = _box(800, 840, 800, 980)
        shift = [_box(200, 240, 200, 320)]
        small = [_box(500, 540, 500, 680)]
        edited = edit_outside_band([lone, supported], shift, small, [outsider, junk])
        self.assertEqual(WITNESS_IOU, 0.5)
        self.assertEqual(len(edited), 2)
        self.assertEqual(int(edited[0][220, 250]), 1)
        self.assertTrue(all(int(mask[30, 40]) == 0 for mask in edited))
        self.assertTrue(any(int(mask[520, 560]) == 1 for mask in edited))
        self.assertTrue(all(int(mask[820, 860]) == 0 for mask in edited))


if __name__ == "__main__":
    unittest.main()
