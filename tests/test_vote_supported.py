"""A vote component that misses the 2048 mask is dropped. One that touches it stays."""

from __future__ import annotations

import unittest

import numpy as np

from filament.paths import IMAGE_SIZE
from filament.rle import decode_rle, encode_rle
from filament.vote_supported import vote_supported


def _rle(y0: int, y1: int, x0: int, x1: int) -> str:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return encode_rle(mask)


class VoteSupportedTest(unittest.TestCase):
    def test_touching_component_kept_and_far_component_dropped(self) -> None:
        parent = [_rle(100, 150, 100, 180)]
        shift = [_rle(100, 150, 100, 180), _rle(400, 460, 400, 480)]
        other = [_rle(100, 150, 100, 180), _rle(400, 460, 400, 480)]
        pieces = [decode_rle(rle) for rle in vote_supported(parent, shift, other)]
        self.assertEqual(len(pieces), 1)
        self.assertEqual(int(pieces[0][120, 140]), 1)
        self.assertEqual(int(pieces[0][430, 440]), 0)


if __name__ == "__main__":
    unittest.main()
