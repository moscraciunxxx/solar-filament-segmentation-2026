"""A 2-of-3 vote drops a lone mask, then a double containment splits the rest."""

from __future__ import annotations

import unittest

import numpy as np

from filament.paths import IMAGE_SIZE
from filament.rle import decode_rle, encode_rle
from filament.vote_contain import vote_then_contain


def _rle(y0: int, y1: int, x0: int, x1: int) -> str:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return encode_rle(mask)


class VoteContainTest(unittest.TestCase):
    def test_lone_mask_dropped_and_two_contained_instances_split(self) -> None:
        parent = [
            _rle(100, 200, 100, 400),
            _rle(700, 740, 700, 780),
        ]
        shift = [_rle(100, 200, 100, 400)]
        other = [
            _rle(120, 170, 120, 190),
            _rle(120, 170, 300, 370),
        ]
        pieces = [decode_rle(rle) for rle in vote_then_contain(parent, shift, other)]
        self.assertEqual(len(pieces), 2)
        self.assertTrue(all(int(mask[720, 740]) == 0 for mask in pieces))
        self.assertNotEqual(int(pieces[0][140, 150]), int(pieces[0][140, 330]))


if __name__ == "__main__":
    unittest.main()
