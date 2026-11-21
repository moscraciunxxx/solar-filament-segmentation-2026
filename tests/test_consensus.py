"""A parent mask stays only when a witness overlaps it."""

from __future__ import annotations

import unittest

import numpy as np

from filament.consensus import keep_if_witnessed
from filament.paths import IMAGE_SIZE
from filament.rle import encode_rle


def _rect(y0: int, x0: int, h: int, w: int) -> str:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[y0 : y0 + h, x0 : x0 + w] = 1
    return encode_rle(mask)


class ConsensusTest(unittest.TestCase):
    def test_unwitnessed_mask_is_dropped(self) -> None:
        source = [_rect(10, 10, 30, 12), _rect(400, 400, 20, 16)]
        witness = [_rect(11, 10, 28, 12)]
        self.assertEqual(keep_if_witnessed(source, witness), [source[0]])

    def test_no_witness_drops_every_mask(self) -> None:
        self.assertEqual(keep_if_witnessed([_rect(10, 10, 8, 8)], []), [])


if __name__ == "__main__":
    unittest.main()
