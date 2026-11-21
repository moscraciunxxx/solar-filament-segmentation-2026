"""A pixel stays when two models mark it and goes when only one does."""

from __future__ import annotations

import unittest

import numpy as np

from filament.paths import IMAGE_SIZE
from filament.rle import encode_rle
from filament.vote import components, vote_map


def _rect(y0: int, x0: int, h: int, w: int) -> str:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[y0 : y0 + h, x0 : x0 + w] = 1
    return encode_rle(mask)


class VoteTest(unittest.TestCase):
    def test_two_of_three_keeps_the_overlap_only(self) -> None:
        shared = _rect(30, 30, 40, 20)
        only_a = _rect(200, 200, 30, 16)
        voted = vote_map([[shared, only_a], [shared], [shared]])
        self.assertEqual(int(voted[40, 35]), 1)
        self.assertEqual(int(voted[210, 205]), 0)
        rles = components(voted)
        self.assertEqual(len(rles), 1)


if __name__ == "__main__":
    unittest.main()
