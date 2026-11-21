"""A blank image takes pixels both donors mark. A non-blank image stays as it is."""

from __future__ import annotations

import unittest

import numpy as np

from filament.empty_fill import fill_empty
from filament.paths import IMAGE_SIZE
from filament.rle import decode_rle, encode_rle


def _rle(y0: int, y1: int, x0: int, x1: int) -> str:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return encode_rle(mask)


class EmptyFillTest(unittest.TestCase):
    def test_nonempty_parent_is_kept_and_blank_needs_both_donors(self) -> None:
        parent = [_rle(10, 20, 10, 20)]
        agreed = _rle(100, 140, 100, 140)
        only_shift = _rle(300, 340, 300, 340)
        self.assertEqual(fill_empty(parent, [agreed], [agreed]), parent)
        filled = [decode_rle(rle) for rle in fill_empty([], [agreed, only_shift], [agreed])]
        self.assertEqual(len(filled), 1)
        self.assertEqual(int(filled[0][120, 120]), 1)
        self.assertEqual(int(filled[0][320, 320]), 0)
        self.assertEqual(fill_empty([], [only_shift], []), [])


if __name__ == "__main__":
    unittest.main()
