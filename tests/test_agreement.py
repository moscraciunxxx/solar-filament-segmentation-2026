"""Agreement keeps a mask only when a witness overlaps it."""

from __future__ import annotations

import unittest

import numpy as np

from filament.agreement import agreeing_rles
from filament.paths import IMAGE_SIZE
from filament.rle import encode_rle


def _rect(y0: int, x0: int, h: int, w: int) -> str:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[y0 : y0 + h, x0 : x0 + w] = 1
    return encode_rle(mask)


class AgreementTest(unittest.TestCase):
    def test_overlap_kept_and_distant_dropped(self) -> None:
        source = [_rect(10, 10, 30, 12), _rect(400, 400, 20, 16)]
        witness = [_rect(12, 11, 28, 12)]
        kept = agreeing_rles(source, witness)
        self.assertEqual(kept, [source[0]])

    def test_empty_witness_keeps_nothing(self) -> None:
        self.assertEqual(agreeing_rles([_rect(10, 10, 8, 8)], []), [])


if __name__ == "__main__":
    unittest.main()
