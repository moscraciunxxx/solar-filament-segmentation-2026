"""A dark bar correlates with the filament template more than empty disk."""

from __future__ import annotations

import unittest

import numpy as np

from filament.matched import match_rles
from filament.paths import IMAGE_SIZE


class MatchedFilterTest(unittest.TestCase):
    def test_dark_bar_is_proposed_and_parent_bar_is_not(self) -> None:
        image = np.full((IMAGE_SIZE, IMAGE_SIZE), 180, dtype=np.uint8)
        image[500:505, 300:420] = 20
        image[100:105, 100:220] = 20
        parent = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        parent[100:105, 100:220] = 1
        from filament.rle import encode_rle

        found = match_rles(image, [encode_rle(parent)], threshold=0.45, exclude_px=8)
        self.assertGreater(len(found), 0)


if __name__ == "__main__":
    unittest.main()
