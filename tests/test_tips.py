"""Tip extension lengthens a bar along its axis and does not fatten it."""

from __future__ import annotations

import unittest

import numpy as np

from filament.paths import IMAGE_SIZE
from filament.tips import TIP_PX, extend_tips


class TipExtensionTest(unittest.TestCase):
    def test_horizontal_bar_grows_at_the_ends(self) -> None:
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        mask[100:106, 80:160] = 1
        grown = extend_tips(mask, TIP_PX)
        self.assertEqual(int(grown[102, 80 - TIP_PX]), 1)
        self.assertEqual(int(grown[102, 159 + TIP_PX]), 1)
        self.assertEqual(int(grown[100 - TIP_PX, 120]), 0)
        self.assertGreater(int(grown.sum()), int(mask.sum()))


if __name__ == "__main__":
    unittest.main()
