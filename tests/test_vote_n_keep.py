"""A vote component that misses the 2048 mask stays only when yolo11n overlaps it."""

from __future__ import annotations

import unittest

import numpy as np

from filament.paths import IMAGE_SIZE
from filament.rle import decode_rle, encode_rle
from filament.vote_n_keep import WITNESS_IOU, vote_n_keep


def _rle(y0: int, y1: int, x0: int, x1: int) -> str:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return encode_rle(mask)


class VoteNKeepTest(unittest.TestCase):
    def test_parent_touch_kept_and_far_piece_needs_n(self) -> None:
        parent = [_rle(100, 150, 100, 180)]
        shift = [
            _rle(100, 150, 100, 180),
            _rle(400, 460, 400, 480),
            _rle(700, 760, 700, 780),
        ]
        other = list(shift)
        small = [_rle(400, 460, 400, 480)]
        pieces = [decode_rle(rle) for rle in vote_n_keep(parent, shift, other, small)]
        self.assertEqual(WITNESS_IOU, 0.5)
        self.assertEqual(len(pieces), 2)
        self.assertTrue(any(int(mask[120, 140]) == 1 for mask in pieces))
        self.assertTrue(any(int(mask[430, 440]) == 1 for mask in pieces))
        self.assertTrue(all(int(mask[730, 740]) == 0 for mask in pieces))


if __name__ == "__main__":
    unittest.main()
